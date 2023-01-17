# RSNA-Breat-cancer-detection
AWS's ML Engineer nano degree capstone project. Breast cancer detection using siamese neural networks.

## Files
RSNA_data_EDA.ipynb. (Exploratory Data Analysis) \
RSNA_dataset_prep.ipynb (Dataset preparation) \
train_model.py (Training script in tesnorflow) \
train_deploy.ipynb (SageMaker SDK for trianing jobs and deployment) \


## Required Dependencies
For preprocessing the dicom and lossless jpeg: \
`
!pip install -qU python-gdcm pydicom pylibjpeg
` \
`
!pip install -U pylibjpeg-openjpeg pylibjpeg-libjpeg
` \
pandas \
Tensorflow==2.3

## Dataset
Data is found in Kaggle's [RSNA Breast Cancer Detection Competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
