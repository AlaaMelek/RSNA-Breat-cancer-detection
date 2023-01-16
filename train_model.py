import numpy as np
import os
import random
import pandas as pd
import argparse
import tensorflow as tf
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
os.environ["CUDA_VISIBLE_DEVICE"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# configs
IMG_SHAPE=(512,512)
INPUT_SHAPE=(512,512,1)
val_split = 0.2
test_split=0.25

# Setup Logger
logger = logging.getLogger('sagemaker')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def parse_function(filename):
    '''
    function to parse strings and read images as tensors
    '''
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SHAPE[0], IMG_SHAPE[1]])
    return image

def preprocess_pairs(image, pair):
    '''
    return images as tuples along with their label
    '''
    return (
        (parse_function(image),
         parse_function(pair[0])),
        pair[1]
    )

def build_train_dataset(imgsPaths, pairsPaths, labels, batch_size):
    '''
    build the train pipeline
    '''
    trainDS = tf.data.Dataset.from_tensor_slices(imgsPaths)
    trainDSPair = tf.data.Dataset.from_tensor_slices((pairsPaths, labels))
    trainDS = tf.data.Dataset.zip((trainDS, trainDSPair))
    trainDS = (trainDS
        .map(preprocess_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(len(imgsPaths))
        .cache()
        #.repeat()
        .batch(batch_size)
    )
    # complete our data input pipeline
    trainDS = (trainDS
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return trainDS

def build_val_dataset(imgsPaths, pairsPaths, labels, batch_size):
    valDS = tf.data.Dataset.from_tensor_slices(imgsPaths)
    valDSPair = tf.data.Dataset.from_tensor_slices((pairsPaths, labels))

    valDS = tf.data.Dataset.zip((valDS, valDSPair))
    valDS = (valDS
        .map(preprocess_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return valDS


def build_siamese_model(inputShape):
    '''
    Function to construct the twin CNNs
    ------
    Params:
    inputShape: (width, height, channels)
    embeddingDim: feature vector size (output of the fully connected layer)
    '''
    # specify the inputs for the feature extractor network
    inputs = Input((inputShape))
    # define the first set of CONV => RELU => POOL
    x = Conv2D(24, (7, 7), padding="same", activation="relu")(inputs) 
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same")(x)
    # second set of CONV => RELU => POOL 
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)
    # two consecutive CONV layers
    x= Conv2D(96,(3,3), padding="same", activation="relu")(x)
    x= Conv2D(96,(3,3), padding="same", activation="relu")(x)
    # final set of CONV => RELU => POOL 
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(x)
    # using global average pooling instead of flatten to reduce no of trainable parameters
    x = GlobalAveragePooling2D()(x)
    # build the model
    model = Model(inputs, outputs=x)
    # return the model to the calling function
    return model

def build_classifier_model(featureShape = 128): #featureShape = 128 in CNN
    '''
    Function to build the fully connected classifier
    -------
    Params:
    Concatenated feature shape
    '''
    # input layer
    inputs = Input(shape=(featureShape,))
    # first fully connected layer ==> RELU ==> dropout
    x = Dense(512, activation="relu")(inputs)
    x = Dropout(0.5)(x)
    # second fully connected layer ==> RELU ==> dropout
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    #outputs = Dense(2, activation="softmax")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    # build the model
    model = Model(inputs, outputs=outputs,name='metric_model')
    # return the model to the calling function
    return model

# build the final network
def build_model():
    print("[INFO] building MatchNet network...")
    imgA = Input(shape=INPUT_SHAPE)
    imgB = Input(shape=INPUT_SHAPE)
    featureExtractor = build_siamese_model(INPUT_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    # concatenate output features
    concat_feats = tf.keras.layers.Concatenate()([featsA, featsB])
    # construct the metric network
    metricNetwork = build_classifier_model(128)
    metric_output = metricNetwork(concat_feats)
    # Final  model
    model = Model(inputs=[imgA, imgB], outputs=metric_output)
    return model

def train(model, trainDS, valDS, batch_size, epochs):

    # start training
    history = model.fit(
        trainDS,
        validation_data=(valDS),
        batch_size=batch_size, 
        epochs= epochs)
    return model

def prepare_paths(train_df, data_dir):

    imgs_paths = []
    pairs_paths = []
    labels = []
    base_dir = os.path.join(data_dir, 'train_images_processed')
    for patient_id in os.listdir(base_dir):
        case_df = train_df[train_df['patient_id']==int(patient_id)]
        case_labels = case_df['cancer'].values
        if 1 in case_labels:
            label=1
        else:
            label=0
        L_MLO =  case_df[case_df['lat_view'] == 'L_MLO']['path'].values[0]
        L_CC =  case_df[case_df['lat_view'] == 'L_CC']['path'].values[0]
        R_MLO =  case_df[case_df['lat_view'] == 'R_MLO']['path'].values[0]
        R_CC = case_df[case_df['lat_view'] == 'R_CC']['path'].values[0]
        imgs_paths.append(os.path.join(base_dir, L_MLO))
        imgs_paths.append(os.path.join(base_dir, L_CC))
        labels.append(label)
        pairs_paths.append(os.path.join(base_dir, R_MLO))
        pairs_paths.append(os.path.join(base_dir, R_CC))
        labels.append(label)

    data = list((zip(imgs_paths,pairs_paths, labels )))
    random.Random(4).shuffle(data)
    img_paths, pair_paths, labels = zip(*data)
    return img_paths, pair_paths, labels

def main(args):
    '''
    Initialize a model by calling the function
    '''
    #prepare paths 
    
    train_df = pd.read_csv(os.path.join(args.data_dir,"new_train.csv"))
    train_df.drop("Unnamed: 0", axis=1, inplace=True)
    train_df['path'] =train_df['patient_id'].astype(str) +'/'+train_df['image_id'].astype(str)+'.png'
    train_df['lat_view'] = train_df['laterality']+"_"+train_df['view']
    img_paths, pair_paths, labels = prepare_paths(train_df, args.data_dir)

    val_index = int(val_split*len(img_paths))
    val_imgs =list(img_paths[:val_index])
    val_pairs = list(pair_paths[:val_index])
    val_labels = list(labels[:val_index])
    train_imgs = list(img_paths[val_index:])
    train_pairs = list(pair_paths[val_index:])
    train_labels = list(labels[val_index:])
    test_index = int(test_split*len(train_imgs))
    test_imgs = train_imgs[:test_index]
    test_pairs = train_pairs[:test_index]
    test_labels = train_labels[:test_index]
    train_imgs = train_imgs[test_index:]
    train_pairs = train_pairs[test_index:]
    train_labels = train_labels[test_index:]
    # tensorflow data pipelines
    trainDS = build_train_dataset(train_imgs,train_pairs, train_labels, args.batch_size)
    valDS = build_val_dataset(val_imgs,val_pairs, val_labels,  args.batch_size)
    
    # build the model 
    model = build_model()
    # compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(loss='binary_crossentropy', optimizer=opt ,metrics=["accuracy", tf.keras.metrics.AUC()])
    
    # dataset pipeline
    
    model=train(model, trainDS, valDS,  args.batch_size, args.epochs)
        
    '''
    Save the trained model
    '''
    #path = os.path.join(args.model_dir, 'siamese')
    model.save('/opt/ml/model/00000001')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # epoch
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="E",
        help="number of epochs to train (default: 20)",
    )
    # batch_size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    # lr
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)"
    )

    # Container environment
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
