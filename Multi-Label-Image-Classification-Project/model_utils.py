from torchvision import models as models
import torch.nn as nn
import torch

def create_model(is_pretrained=True, num_outputs=14, model_name='resnet50'):
    
    model = None
    if model_name.lower() == 'resnet50':
        
        model = models.resnet50(progress=True, pretrained=is_pretrained)
#         model.fc = nn.Sequential(
#                 nn.Linear(2048, 1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, num_outputs),
#                 nn.Sigmoid())
        model.fc = nn.Sequential(
                    nn.Linear(2048, 1411),
                    nn.ReLU(),
                    nn.Linear(1411, num_outputs),
                    nn.Sigmoid())
        
    elif model_name.lower() == 'resnet34':
        
        model = models.resnet34(progress=True, pretrained=is_pretrained)
        model.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_outputs),
                nn.Sigmoid())
        
    elif model_name.lower() == 'densenet':
        
        model = models.densenet121(progress=True, pretrained=is_pretrained)
        model.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_outputs),
                nn.Sigmoid())
        
    return model

def freeze_model_parameters(model):
    
    for params in model.parameters():
        params.requires_grad = False
    
    model_name = model.__class__.__name__
    
    if model_name.lower() == 'resnet':
        for params in model.fc.parameters():
            params.requires_grad = True           # Unfreezing the last fully connected layers
    
    elif model_name.lower() == 'densenet':
        for params in model.classifier.parameters():
            params.requires_grad = True           # Unfreezing the last fully connected layers


def unfreeze_model_parameters(model):
    
    for params in model.parameters():
        params.requires_grad = True


def load_model(model, path):
    
    model_checkpoint = torch.load(path)
    model.load_state_dict(model_checkpoint['model_state_dict'], strict = False)
    
    return model

def load_model_for_prediction(model, path):
    
    model_checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint['model_state_dict'], strict = False)
    
    return model

def build_model_for_uptuna(trial, model, num_outputs=14, model_name='resnet50'):
        
    # Suggest the number of layers of neural network model
    n_layers = trial.suggest_int("n_layers", 1, 4)
    
    if model_name.lower() == 'resnet50':
                
        layers = generate_model_layers(trial=trial, in_features=2048, n_layers=n_layers, num_outputs=num_outputs)
        model.fc = nn.Sequential(*layers)
        
    elif model_name.lower() == 'resnet34':
                
        layers = generate_model_layers(trial=trial, in_features=512, n_layers=n_layers, num_outputs=num_outputs)
        model.fc = nn.Sequential(*layers)
        
    elif model_name.lower() == 'densenet':
                
        layers = generate_model_layers(trial=trial, in_features=1024, n_layers=n_layers, num_outputs=num_outputs)
        model.classifier = nn.Sequential(*layers)
        
    return model


def generate_model_layers(trial, in_features, n_layers, num_outputs):
    
    layers = []

    for i in range(n_layers):
        
        # Suggest the number of units in each layer
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 2048)
        
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        in_features = out_features
        
    layers.append(nn.Linear(in_features, num_outputs))
    layers.append(nn.Sigmoid())
    
    return layers


def apply_optuna_learned_fully_connected_layers(model, num_outputs):
        
        model.fc = nn.Sequential(
            nn.Linear(2048, 1411),
            nn.ReLU(),
            nn.Linear(1411, num_outputs),
            nn.Sigmoid())
        
        return model

    

    

    

    

    

    

    

    

