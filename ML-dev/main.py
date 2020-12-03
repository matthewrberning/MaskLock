import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#logging
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# from data_loader import create_dataloaders
#//////////////////////////////////////////
from _data_loader import build_dataloaders

from _model import trainable_params, create_model, print_model_params

#metrics
from sklearn.metrics import roc_auc_score

# arguments
import argparse

# track time of run
from time import time, strftime, gmtime

import time

#progress bar
from tqdm import tqdm

def train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, writer, device):
    #put model into training mode
    model.train()

    #set training loss var for this epoch
    train_loss = 0

    #setup progressbar and dl
    progress_bar = tqdm(train_dl, total=int(len(train_dl)), desc='Training Progress: ')
    
    #reset of gradients at begining of epoch
    optimizer.zero_grad()

    #tracking vars for roc/acc metric
    total = 0
    corrects_count = 0

    all_labels = []
    all_predictions = []
    
    for step, data in enumerate(progress_bar):
        #colledt data
        inputs = data['image']
        labels = data['label'].view(-1)
        
        #send to device
        inputs = inputs.cuda(device=0)
        labels = labels.cuda(device=0)
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        
        #reset grads to zero at the begining of each mini-batch
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            
            _, predictions = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            
            corrects_count += (predictions == labels).sum().item()
            
            loss = criterion(outputs, labels)

            loss.backward() #backpropagating the loss

            optimizer.step() #stepping down the gradient

        #tracking for roc-auc
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        
        #collecting the losses
        train_loss += loss.item()
               
        progress_bar.set_postfix(loss=train_loss / (step + 1), acc=corrects_count / total)
    
    train_losses = train_loss / len(train_dl)
    train_accs = corrects_count / total

    #generate an ROC-AUC for the trainig epoch
    try:
        train_roc = roc_auc_score(all_labels, all_predictions)
        writer.add_scalar(tag='ROC-AUC/train_roc', scalar_value=train_roc, global_step=epoch)
    except:
        print("roc screwed up")
        pass

    #tensorboard||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    writer.add_scalar(tag='Loss/train_losses', scalar_value=train_losses, global_step=epoch)
    writer.add_scalar(tag='Accuracy/train_accs', scalar_value=train_accs, global_step=epoch)
    
    print(f'\nEpoch {epoch}: train loss={train_losses:.4f} | train accuracy={train_accs:.4f}')

def validate(epoch, model, val_dl, criterion, writer, device):
    #set model to evaluation mode
    model.eval()

    #set vars for metrics tracking
    val_loss = 0
    correct_count = 0
    total = 0

    all_labels = []
    all_predictions = []

    #////////////////////////
    #setup progressbar and dl
    progress_bar = tqdm(val_dl, total=int(len(val_dl)), desc='Validation Epoch')
    #/////////////////////////

    # for data in val_dl:

    #///////////////////////
    for step, data in enumerate(progress_bar):
    #//////////////////////

        #collect data
        inputs = data['image']
        labels = data['label'].view(-1)

        #send to device

        # inputs = inputs.to(device)
        # labels = labels.to(device)
        inputs = inputs.cuda(device=0) 
        labels = labels.cuda(device=0)


        with torch.no_grad():
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct_count += (predictions == labels).sum().item()
            val_loss += criterion(outputs, labels)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    #calculate metrics
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    #calculate ROC-AUC
    try:
        val_roc = roc_auc_score(all_labels, all_predictions)
        writer.add_scalar(tag='ROC-AUC/val_roc', scalar_value=val_roc, global_step=epoch)
    except:
        print("roc screwed up")
        pass
    val_acc = correct_count / total
    val_losses = val_loss / len(val_dl)

    #tensorboard|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    writer.add_scalar(tag='Loss/val_loss', scalar_value=val_losses, global_step=epoch)
    writer.add_scalar(tag='Accuracy/val_accs', scalar_value=val_acc, global_step=epoch)


    print(f'\t val loss={val_losses:.4f} | val accuracy={val_acc:.4f} | ')
    #Change this back when ROC is fixed
    # print(f'\t val loss={val_losses:.4f} | val acc={val_acc:.4f} | '
    #   f'val ROC={val_roc:.4f}')

    return val_acc


def train(model_name, n_epochs, lr, batch_size, dataset_path):

    # Fix random seed if needed
    # torch.manual_seed(30)
    # np.random.seed(30)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("...device being used: ", device)

    print('...building dataset')
    #make the dataloaders
    #//////////////////////////////////////////////////////////////////////////WORK ON DATALOADER
    # train_dl, val_dl = create_dataloaders(dataset_path, batch_size, device)
    train_dl, val_dl = build_dataloaders(dataset_path, batch_size)

    print('...assembling model')
    #set up the loss fucntion
    criterion = nn.CrossEntropyLoss()
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    # weight = torch.tensor([0, 3]) #weighting for the positive class
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    #pull resNet18 model with trainable layers
    if model_name == "resnet18":
        model = create_model()

    _ = print_model_params(model)
    params = trainable_params(model)

    # Create optimizer and learning rate 
    optimizer = optim.Adam(params, lr=lr)


    #send model to device
    model = model.to(device)


    print('Training start..')

    # #tensorboard setup||||||||||||||||||||||||||||||
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S")
    writer = SummaryWriter(f'./runs/{current_time}')

    #set best metric for training run to zero
    best_metric = 0.

    os.makedirs('./checkpoints/', exist_ok=True)


    for epoch in range(n_epochs):

        print(f"\nSTARTING EPOCH: {epoch} OF: {n_epochs}\n")

        #send to training step
        train_one_epoch(epoch, model, train_dl, lr, optimizer, criterion, writer=writer, device=device)

        #send to validation step and return validation accuracy
        current_metric = validate(epoch, model, val_dl, criterion, writer=writer, device=device)

        #get current time
        current_time = time.strftime("%Y-%m-%d-%H_%M_%S")

        if current_metric >= best_metric:
            print(f'\n\n>>>>>>>>\n\n    saving best model (Validation Accuracy: {current_metric:.4f})')
            print(f"                      previous best Validation Accuracy: {best_metric:.4f}")
            print(f"\n    model saved at: 'checkpoints/{current_time}--best_model.pth'\n\n>>>>>>>>\n\n")

            checkpoint = {'model': model,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, f'checkpoints/{current_time}--best_model.pth')
            best_metric = current_metric

    #close the tensorboard loggers
    writer.close()



#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////

def get_args():
    parser = argparse.ArgumentParser(description="------------")
    parser.add_argument('--model_name', type=str, default='resnet18', required=False,
                        help='specify the model name')
    #/////////////////////////////////testing defaults -> CHANGE THESE
    parser.add_argument('--num_epochs', type=int, default=20, required=False,
                        help='specify the number of epochs')
    #/////////////////////////////////testing defaults -> CHANGE THESE
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='specify the batch size')

    parser.add_argument('--verbose', '--v', type=bool, default=False, required=False,
                        help='verbosity, on or off -> bool')

    return parser.parse_args()


if __name__ == '__main__':

    #get arguments
    args = get_args()

    # Get time of run
    curr_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    print(f"\n\nTRAINING MODEL\nmodel name: {args.model_name}\nnumber of epochs: {args.num_epochs}\nbatch size:{args.batch_size}")

    learning_rate = 1.0e-6

    path = "E:/One Drive/OneDrive/Mainproject/MaskLockData/dataset2/"

    #pass to training fucntion
    train(model_name=args.model_name, n_epochs=args.num_epochs, lr=learning_rate, batch_size=args.batch_size, dataset_path=path)

    print("\n\n\n...done!\n")


