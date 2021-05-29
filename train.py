import argparse

import torch
from torch import nn, optim

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os.path import isdir



def get_dataloders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    means = [0.485, 0.456, 0.40]
    deviations = [0.229, 0.224, 0.225]

    data_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means,deviations)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(means,deviations)
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(means,deviations)
        ])
    }
    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(test_dir , transform=data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    test_loader  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    
    return train_loader, valid_loader , test_loader ,  image_datasets['train']
    
    
def validation(model, valid_loader, criterion,device):
    model.to(device)
        
    valid_loss = 0
    accuracy = 0
    for data in valid_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
                
        output = model.forward(images)
        loss = criterion(output, labels).data[0];
        valid_loss += loss
        ps = torch.exp(output)
        equality = labels.data == ps.max(1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


def network_trainer(train_loader, valid_loader,images_train_datasets,arch, hidden_units, learning_rate,epochs,device):
    valid_length = len(valid_loader)

    print_every = 30
    steps = 0
    
    model = eval("models.{}(pretrained=True)".format(arch))
    
    for params in model.parameters():
        params.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu', nn.ReLU()), 
    ('fc2', nn.Linear(hidden_units, 102)),
    ('drop', nn.Dropout(p=0.5)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    for e in range(epochs):
        print("Epoch: {}/{}".format(e+1, epochs))

        model.train()

        train_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            
            inputs = inputs.to(device)
            labels = labels.to(device)
   


            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion,device)

                print(
                      "Training Loss: {:.3f} -- ".format(train_loss/print_every),
                      "Valid Loss: {:.3f} -- ".format(valid_loss/valid_length),
                      "Valid Accuracy: {:.3f}".format(accuracy/valid_length))

                train_loss = 0

                model.train()
    save_model(model,arch,images_train_datasets)
            
    return model
def  save_model(model_trained,arch,images_train_datasets):
    model_trained.class_to_idx = images_train_datasets.class_to_idx

    checkpoint = {
              'class_to_idx': model_trained.class_to_idx,
              'classifier'  : model_trained.classifier, 
              'state_dict'  : model_trained.state_dict(),
              'arch'        : arch            
                }

    torch.save(checkpoint, 'checkpoint.pth')
    print("Done. model is trained and saved")
    
    
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
        
    return device



def main():
    train_loader, valid_loader, test_loader , images_train_datasets = get_dataloders('flowers')

    parser = argparse.ArgumentParser(description='Train Dataset')


    parser.add_argument('--arch', type=str, \
        help='Models architeture. Default is vgg16.')
    parser.add_argument('--learning_rate', type=float, \
        help='Learning rate. Default is 0.001')
    parser.add_argument('--hidden_units', type=int, \
        help='Hidden units. Default is 256')
    parser.add_argument('--epochs', type=int, \
        help='Number of epochs. Default is 5')
    parser.add_argument('--gpu', action='store_true', \
        help='Use GPU for inference if available')

    
    args, _ = parser.parse_known_args()

    
    arch = 'vgg16'
    if args.arch:
        arch = args.arch

    learning_rate = 0.001
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 256
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 1
    if args.epochs:
        epochs = args.epochs

    device = check_gpu(args.gpu);
            
    model = network_trainer(train_loader, valid_loader,images_train_datasets,arch, hidden_units, learning_rate,epochs,device)

   
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    