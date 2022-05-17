import torch
from dataset import ImageDataset
from model_utils import create_model, freeze_model_parameters, unfreeze_model_parameters, load_model
from train_validate import train_, validate_
from test_predict import test_, predict_, batch_predict_

import numpy as np
import pandas as pd
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
from torch.optim import lr_scheduler
import copy
from helper_functions import download_images_from_gcs, download_model_from_gcs, is_file_exists_on_google_cloud_storage, delete_model_from_gcs
from google.cloud import storage

# Initialise a client
storage_client = storage.Client(project='xxxxxxxxxx') # Your GCP Project ID here

# Upload files to Google Cloud Storage Bucket
def upload_files(bucket_name, source_folder):
    bucket = storage_client.get_bucket(bucket_name)
    for filename in os.listdir(source_folder):
        blob = bucket.blob("model/" + filename)
        blob.upload_from_filename(source_folder + '/' + filename)
        
print('!!!Downloading images from Google Cloud Storage to Container images folder!!!')        
download_images_from_gcs()

batch_size = 50
epochs = 2
learning_rate = 0.03
criterion = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataframe = pd.read_csv('gs://multi-label-image-classification-bucket/input/train.csv')

train_df_path = 'gs://multi-label-image-classification-bucket/input/train_df.csv'
validation_df_path = 'gs://multi-label-image-classification-bucket/input/validation_df.csv'
test_df_path = 'gs://multi-label-image-classification-bucket/input/test_df.csv'

is_train_file_exists = is_file_exists_on_google_cloud_storage(file_name='input/train_df.csv', bucket_name='multi-label-image-classification-bucket')
is_validation_file_exists = is_file_exists_on_google_cloud_storage(file_name='input/validation_df.csv', bucket_name='multi-label-image-classification-bucket')
is_test_file_exists = is_file_exists_on_google_cloud_storage(file_name='input/test_df.csv', bucket_name='multi-label-image-classification-bucket')

if is_train_file_exists & is_validation_file_exists & is_test_file_exists:
    
    train_df = pd.read_csv(train_df_path)
    validation_df = pd.read_csv(validation_df_path)
    test_df = pd.read_csv(test_df_path)
    
    print('!!!Files exists!!!')
    
else:
    # DATA PREPROCESSING STEP

    # Getting dataframe of all labels
    genres_df = dataframe.columns.values[2:]

    filter_label_columns_list = []
    for genre in genres_df:
        items_available_in_dataset = dataframe[genre].value_counts()[1] # Getting number of True values per genre

        # Creating list of label columns that has number of TRUE values less than 350
        if items_available_in_dataset < 350:
            filter_label_columns_list.append(genre)

    filtered_df = dataframe.drop(filter_label_columns_list, axis=1) # Droping filtered label columns

    X = filtered_df['Id'].values
    y = filtered_df.drop(['Id','Genre'], axis=1).values

    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

    for train_index, valid_index in msss.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    for validation_index, test_index in msss.split(X_valid, y_valid):
        X_Validation, X_test = X_valid[validation_index], X_valid[test_index]
        y_validation, y_test = y_valid[validation_index], y_valid[test_index]

    train_dataset_array = np.concatenate([np.reshape(X_train, (-1, 1)), y_train], axis=1)
    validation_dataset_array = np.concatenate([np.reshape(X_Validation, (-1, 1)), y_validation], axis=1)
    test_dataset_array = np.concatenate([np.reshape(X_test, (-1, 1)), y_test], axis=1)

    train_df = pd.DataFrame(data=train_dataset_array, columns=filtered_df.drop('Genre', axis=1).columns.values)
    validation_df = pd.DataFrame(data=validation_dataset_array, columns=filtered_df.drop('Genre', axis=1).columns.values)
    test_df = pd.DataFrame(data=test_dataset_array, columns=filtered_df.drop('Genre', axis=1).columns.values)

    train_df.to_csv(train_df_path, index=False)
    validation_df.to_csv(validation_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)
    
    print('!!!Files created and saved!!!')
    
    
train_data = ImageDataset(train_df, is_train=True, is_test=False)
validation_data = ImageDataset(validation_df, is_train=False, is_test=False)
test_data = ImageDataset(test_df, is_train=False, is_test=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

num_labels = len(train_data[0][1])

model = create_model(is_pretrained=True, num_outputs=num_labels, model_name='resnet50').to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
best_accuracy = 0.0
best_f1_score = 0.0

model_path = 'gs://movie-classifier-model-bucket/model/best-model.pth'
is_model_exists = is_file_exists_on_google_cloud_storage(file_name='model/best-model.pth', bucket_name='movie-classifier-model-bucket')

if is_model_exists:
    
    print("Downloading the previously saved model from Google Cloud Storage Bucket")
    
    model_path = './model/best-model.pth'
    download_model_from_gcs(destination_path=model_path)
    
    #load the the best model
    print("Loading the previously saved best model")
    best_model_checkpoint = torch.load(model_path)

    #load model weights state_dict
    model = load_model(model, model_path)
    optimizer.load_state_dict(best_model_checkpoint['optimizer_state_dict'])
    
    metrics = best_model_checkpoint['validation_metrics']
    best_accuracy = metrics['multilabel_accuracy_per_epoc']
    best_f1_score = metrics['multilabel_f1_score_per_epoc']
    print(f"Best Accuracy of previous model is: {best_accuracy}")
    print(f"Best F1-score of previous model is: {best_f1_score}")
else:
    print("Model not found. Train a new resnet model from scratch.")

# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

import time
start_time = time.time()

train_loss = []
validation_loss = []

for epoc in range(epochs):
    
    print(f'Epoc: {epoc+1} of {epochs}')
    
    epoc_start_time = time.time()
        
    unfreeze_model_parameters(model)
         
    train_metrics = train_(model, train_loader, optimizer, criterion, exp_lr_scheduler, train_data, device)
    validation_metrics = validate_(model, validation_loader, criterion, validation_data, device)
    
    epoc_end_time = time.time()
    total_epoc_time = epoc_end_time - epoc_start_time
    print(f'Epoc {epoc+1} took {(total_epoc_time/60): .3f} minutes\n')

    train_loss.append(train_metrics['train_loss_per_epoc'])
    validation_loss.append(validation_metrics['validation_loss_per_epoc'])
    
    genres = train_df.columns.values[1:]
    train_accuracies_list = train_metrics['accuracies_list']
    validation_accuracies_list = validation_metrics['accuracies_list']
    
    traning_metrics = f"""
    Traning Mertics:

    Loss: {train_metrics['train_loss_per_epoc']: .4f}
    Accuracy: {train_metrics['multilabel_accuracy_per_epoc']: .4f}
    Precision: {train_metrics['multilabel_precision_per_epoc']: .4f}
    Recall: {train_metrics['multilabel_recall_per_epoc']: .4f}
    F1-score: {train_metrics['multilabel_f1_score_per_epoc']: .4f}
    """
    print(traning_metrics)
    print('Accuracies per genre:\n')
    for i in range(len(genres)):
        print(f'{genres[i]}: {(train_accuracies_list[i] * 100): .2f}')
    
    valid_metrics = f"""
    \nValidation Mertics:

    Loss: {validation_metrics['validation_loss_per_epoc']: .4f}
    Accuracy: {validation_metrics['multilabel_accuracy_per_epoc']: .4f}
    Precision: {validation_metrics['multilabel_precision_per_epoc']: .4f}
    Recall: {validation_metrics['multilabel_recall_per_epoc']: .4f}
    F1-score: {validation_metrics['multilabel_f1_score_per_epoc']: .4f}
    """
    print(valid_metrics)
    print('Accuracies per genre:\n')
    for i in range(len(genres)):
        print(f'{genres[i]}: {(validation_accuracies_list[i] * 100): .2f}')
        
    print('\n')
    
    current_f1_score = validation_metrics['multilabel_f1_score_per_epoc']
    
    # save the Best Model to disk after every epoc
    if best_f1_score == 0 or (current_f1_score > best_f1_score):
        print('Updating best f1_score: previous best = {:.3f} new best = {:.3f}'.format(best_f1_score,
                                                                                     current_f1_score))
        best_f1_score = current_f1_score
        
        is_model_exists = is_file_exists_on_google_cloud_storage(file_name='model/best-model.pth', bucket_name='movie-classifier-model-bucket')
        
        print('After is_model_exists')
        if is_model_exists:
            print('Model exists')
            delete_model_from_gcs() #Deleting previously saved best-model.pth because GCS doesnot allow to overwrite file with same name
            print('Deleting the previously saved model')
            
        torch.save({
            'epoch': epoc+1,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'traning_metrics': train_metrics,
            'validation_metrics': validation_metrics
            }, './model/best-model.pth')
        
        print('New best model saved to container model folder')
        print('Now uploading the best model to google cloud storage model bucket')
        
        upload_files('movie-classifier-model-bucket', './model')
        print('!!!File uploaded successfully!!!')           
    
    
current_time = time.time()
total = current_time - start_time
print(f'Traning took {(total/60): .3f} minutes')