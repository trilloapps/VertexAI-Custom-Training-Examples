from tqdm import tqdm
from multilabel_metrics import get_multilabel_accuracy, get_multilabel_precision_recall_f1score 
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import cv2
import pandas as pd
from model_utils import create_model, load_model_for_prediction
from helper_functions import is_file_exists_on_google_cloud_storage, download_model_from_gcs
import base64
from PIL import Image
import io

def test_(model, dataloader, test_data, genres, device):
    print('Testing')
    model.eval()
    
    num_labels = 0
    
    targets_list = []
    predictions_list = []
    predicted_genres_list = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data)/dataloader.batch_size)):
            
            image, label = data[0].to(device), data[1].to(device)
            num_labels = len(label[0])
            
            for item in range(len(label)):
                target_indices = [i for i in range(len(label[item])) if label[item][i] == 1]
                targets_list.append(convert_multilabels_to_binary_form(len(label[item].cpu().data.numpy()), target_indices))
        
            outputs = model(image)            
            outputs = outputs.detach().cpu()
            
            for item in range(len(outputs)):
                outputs_indices = np.argsort(outputs[item])[-3:]
                predictions_list.append(convert_multilabels_to_binary_form(len(outputs[item].cpu().data.numpy()), outputs_indices))
                predictions = convert_multilabels_to_binary_form(len(outputs[item].cpu().data.numpy()), outputs_indices)
                prediction_indices = [i for i in range(len(predictions)) if predictions[i] == 1]                
                
                predicted_genres = []
                for i in range(len(prediction_indices)):
                     predicted_genres.append(genres[prediction_indices[i]])
            
                predicted_genres_list.append(predicted_genres)

        multilabel_accuracy, accuracies_list = get_multilabel_accuracy(num_labels, targets_list, predictions_list)
        multilabel_precision, multilabel_recall, multilabel_f1_score = get_multilabel_precision_recall_f1score(targets_list, predictions_list)

        metrics_dict = {
            'multilabel_accuracy': multilabel_accuracy,
            'multilabel_precision': multilabel_precision,
            'multilabel_recall': multilabel_recall,
            'multilabel_f1_score': multilabel_f1_score,
            'accuracies_list': accuracies_list
        }

        return metrics_dict, predicted_genres_list


def predict_(b64_string):
    print('Predicting')
    
    model, genres, device = get_saved_model_geners_and_device()
    
    model.eval()
        
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4321, 0.3840, 0.3556], [0.3130, 0.2917, 0.2786])
            ])
    
    with torch.no_grad():
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(b64_string, "utf-8"))))
        #image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image_detach = image.detach().clone()

        output = model(image_detach.type(torch.float32).view(1,3,224,224).to(device))            
        output = output.detach().cpu()[0]
        
        output_indices = np.argsort(output)[-3:]
        
        predictions = convert_multilabels_to_binary_form(len(output.cpu().data.numpy()), output_indices)
        prediction_indices = [i for i in range(len(predictions)) if predictions[i] == 1]                

        predicted_genres = []
        for i in range(len(prediction_indices)):
             predicted_genres.append(genres[prediction_indices[i]])

        return predicted_genres


def batch_predict_(image_paths):
    print('Predicting')
    
    model, genres, device = get_saved_model_geners_and_device()
    
    model.eval()
    
    predicted_genres_list = []
    
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4321, 0.3840, 0.3556], [0.3130, 0.2917, 0.2786])
            ])
    
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            image_detach = image.detach().clone()
               
            outputs = model(image_detach.type(torch.float32).view(1,3,224,224).to(device))            
            outputs = outputs.detach().cpu()
            
            for item in range(len(outputs)):
                outputs_indices = np.argsort(outputs[item])[-3:]
                predictions = convert_multilabels_to_binary_form(len(outputs[item].cpu().data.numpy()), outputs_indices)
                prediction_indices = [i for i in range(len(predictions)) if predictions[i] == 1]                
                
                predicted_genres = []
                for i in range(len(prediction_indices)):
                     predicted_genres.append(genres[prediction_indices[i]])
            
                predicted_genres_list.append(predicted_genres)

        return predicted_genres_list


def convert_multilabels_to_binary_form(num_classes, indices):
    
    convert_to_binary = []
    
    for i in range(num_classes):
    
        if i in indices:
            convert_to_binary.append(1)
        else:
            convert_to_binary.append(0)
    
    return convert_to_binary


def get_saved_model_geners_and_device():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_df = pd.read_csv('/home/model-server/test_df.csv')
    genres = test_df.columns.values[1:]
    
    num_labels = len(genres)

    model_path = '/home/model-server/model/best-model.pth'

    #load model weights state_dict
    model = create_model(is_pretrained=True, num_outputs=num_labels, model_name='resnet50').to(device)
    model = load_model_for_prediction(model, model_path)
    
    return model, genres, device

