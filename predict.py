from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
from torchvision import datasets, transforms, models
import shutil, argparse, json



    
def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    
    model = eval("models.{}(pretrained=True)".format(checkpoint['arch']))
    
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model




def process_image(image):

    img_processing = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
        ])
    
    image = img_processing(Image.open(image))
    
    return image

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
        
    return device

def predict(image_path, model,device,topk):
  
    model.eval()
    
    model.to(device)
    
    img = process_image(image_path)
    if str(device) == 'cpu':
     tensor = img.unsqueeze_(0).float()
    else:
     tensor = img.unsqueeze_(0).float().cuda()
        
    with torch.no_grad():
        output = model.forward(tensor)
        
    probabilities = F.softmax(output.data,dim=1)
    
    
    return probabilities.topk(topk)

def get_class_names(probabilities,json_path):
    
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    classes_names = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
    
    return classes_names
    

def main():
    parser = argparse.ArgumentParser(description='predict')

    
    parser.add_argument('--ipath', type=str, \
        help='ImagePath. default is flowers/test/70/image_05331.jpg ')
    
    parser.add_argument('--cppath', type=str, \
        help='checkpoint path. Default is checkpoint.pth')

    
    parser.add_argument('--jpath', type=str, \
        help='json path to map category names. default is cat_to_name.json')
    
    parser.add_argument('--topk',type=int, \
        help='top k probabilities. default is 1')
    
    parser.add_argument('--gpu', action='store_true', \
        help='use gpu')
    

    
    args, _ = parser.parse_known_args()

    
    ipath = 'flowers/test/70/image_05331.jpg'
    if args.ipath:
        ipath = args.ipath

    cppath = 'checkpoint.pth'
    if args.cppath:
        cppath = args.cppath

    topk = 1
    if args.topk:
        topk = args.topk

    jpath = 'cat_to_name.json'
    if args.jpath:
        jpath = args.jpath
    
    device = check_gpu(args.gpu);

    model = load_checkpoint(cppath)
    
    probabilities = predict(ipath,model,device,topk)
    p = np.array(probabilities[0][0])

    
    classes_names = get_class_names(probabilities,jpath)
    
    print("the class name is  : ",classes_names[0])
    print("the probability is : ",p[0])
    
    if topk > 1 :
        print("the classes names are  : ", classes_names)
        print("the probabilities are : ", p)

    

    
    
    
    



if __name__ == '__main__':
    main()