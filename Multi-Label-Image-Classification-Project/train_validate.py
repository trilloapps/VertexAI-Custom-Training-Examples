from tqdm import tqdm
from multilabel_metrics import get_multilabel_accuracy, get_multilabel_precision_recall_f1score 
import numpy as np
import torch

def train_(model, dataloader, optimizer, criterion, scheduler, train_data, device):
    print('Traning')
    model.train()
    
    batches = 0
    train_running_loss = 0.0
    num_labels = 0
    
    targets_list = []
    predictions_list = []
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        batches += 1
        
        image, label = data[0].to(device), data[1].to(device)
        num_labels = len(label[0])
        
        for item in range(len(label)):
            target_indices = [i for i in range(len(label[item])) if label[item][i] == 1]
            targets_list.append(convert_multilabels_to_binary_form(len(label[item].cpu().data.numpy()), target_indices))
        
        optimizer.zero_grad()
        outputs = model(image)
        
        loss = criterion(outputs, label)
        train_running_loss += loss.item()
        
        outputs = outputs.detach().cpu()
        
        for item in range(len(outputs)):
            outputs_indices = np.argsort(outputs[item])[-3:]
            predictions_list.append(convert_multilabels_to_binary_form(len(outputs[item].cpu().data.numpy()), outputs_indices))
        
        loss.backward()
        optimizer.step()
#         scheduler.step()
        # if batches == 1:
        #     break
        
    multilabel_accuracy, accuracies_list = get_multilabel_accuracy(num_labels, targets_list, predictions_list)
    multilabel_precision, multilabel_recall, multilabel_f1_score = get_multilabel_precision_recall_f1score(targets_list, predictions_list)

    train_loss = train_running_loss / batches
    
    metrics_dict = {
        'train_loss_per_epoc': train_loss,
        'multilabel_accuracy_per_epoc': multilabel_accuracy,
        'multilabel_precision_per_epoc': multilabel_precision,
        'multilabel_recall_per_epoc': multilabel_recall,
        'multilabel_f1_score_per_epoc': multilabel_f1_score,
        'accuracies_list': accuracies_list
    }
    
    return metrics_dict


def validate_(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    
    batches = 0
    validation_running_loss = 0.0
    num_labels = 0
    
    targets_list = []
    predictions_list = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            batches += 1
            
            image, label = data[0].to(device), data[1].to(device)
            num_labels = len(label[0])
            
            for item in range(len(label)):
                target_indices = [i for i in range(len(label[item])) if label[item][i] == 1]
                targets_list.append(convert_multilabels_to_binary_form(len(label[item].cpu().data.numpy()), target_indices))
        
            outputs = model(image)
            
            loss = criterion(outputs, label)
            validation_running_loss += loss.item()
            
            outputs = outputs.detach().cpu()
            
            for item in range(len(outputs)):
                outputs_indices = np.argsort(outputs[item])[-3:]
                predictions_list.append(convert_multilabels_to_binary_form(len(outputs[item].cpu().data.numpy()), outputs_indices))
            
            # if batches == 1:
            #     break

        multilabel_accuracy, accuracies_list = get_multilabel_accuracy(num_labels, targets_list, predictions_list)
        multilabel_precision, multilabel_recall, multilabel_f1_score = get_multilabel_precision_recall_f1score(targets_list, predictions_list)

        validation_loss = validation_running_loss / batches

        metrics_dict = {
            'validation_loss_per_epoc': validation_loss,
            'multilabel_accuracy_per_epoc': multilabel_accuracy,
            'multilabel_precision_per_epoc': multilabel_precision,
            'multilabel_recall_per_epoc': multilabel_recall,
            'multilabel_f1_score_per_epoc': multilabel_f1_score,
            'accuracies_list': accuracies_list
        }

        return metrics_dict


def convert_multilabels_to_binary_form(num_classes, indices):
    
    convert_to_binary = []
    
    for i in range(num_classes):
    
        if i in indices:
            convert_to_binary.append(1)
        else:
            convert_to_binary.append(0)
    
    return convert_to_binary




