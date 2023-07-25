# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:39:21 2023

@author: khelv
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm import create_model

def main(args):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomResizedCrop((100,120), scale=(0.8,1.0), ratio=(0.9,1.1)), # limited resize crop => keep the crack in the augmented image
        transforms.RandomRotation((-80,80)), # arbitrarily chosen angles: you can play with the angle in order to generate more variation in augmented images
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = ImageFolder(args.training_rep, transform=train_transform)
    test_dataset = ImageFolder(args.test_rep, transform=test_transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model
    model = create_model(args.model, pretrained=args.pretrained, num_classes=len(train_dataset.classes)).to(device)

    # Calculate the number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

    # Evaluate model on test set
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}')
    print(f'Confusion matrix:\n{cm}')

    # Save log and model state dict
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, 'log.txt'), 'w') as f:
            f.write(f'Model: {args.model}, Number of parameters: {num_params}\n')
            f.write(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}\n')
            f.write(f'Confusion matrix:\n{cm}')

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))

if __name__ == '__main__':
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Train a classification model from the timm library')
    parser.add_argument('--training_rep', type=str, help='Path to the training dataset', default=r'./FLYD-C/Train_set')
    parser.add_argument('--test_rep', type=str, help='Path to the test dataset', default=r'./FLYD-C/Test_set')
    parser.add_argument('--model', type=str, help='Name of the timm model to use',default='vgg13')
    parser.add_argument('--pretrained', type=str, help='Load the pretrained model or the randomly initialized model',default=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train the model', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch_size feeding the network for training and evaluation', default=16)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer', default=1e-4)
    parser.add_argument('--output_dir', type=str, help='Path to the directory to save the log and model state dict', default=None)

    # Parse command-line arguments
    args = parser.parse_args()

    main(args)