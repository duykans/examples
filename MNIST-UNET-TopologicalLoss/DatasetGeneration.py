import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
import os

#Parameters
dataset_size = 15000
cmap = "gray"
size = 64

#MNIST with gaussian noise
class GaussianNoiseTransform(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + "(mean = {0}, std = {1})".format(self.mean, self.std)

def generate_raw_dataset_split(data, ground_truth, data_split=0.8, cmap="gray", path="./", dataset_name= "cm"):
    if(path[-1] != "/" or dataset_name[0] != "/"):
        path = path + "/"
        
    if(dataset_name[-1] != "/"):
        final_path = path + dataset_name + "/"
    else:
        final_path = path + dataset_name
    
    if not os.path.exists(final_path):
        os.makedirs(final_path + "train/data/images") 
        os.makedirs(final_path + "train/ground_truth/images") 
        os.makedirs(final_path + "test/data/images")
        os.makedirs(final_path + "test/ground_truth/images")
        
    for i in range(int(len(data) * data_split)):
        matplotlib.image.imsave(final_path + "train/data/images/" + str(i) + ".png", data[i], cmap=cmap)
        matplotlib.image.imsave(final_path + "train/ground_truth/images/" + str(i) + ".png", ground_truth[i], cmap=cmap)
        
    for i in range(int(len(data) * data_split)+ 1 ,  len(data) ):
        matplotlib.image.imsave(final_path + "test/data/images/" + str(i) + ".png", data[i], cmap=cmap)
        matplotlib.image.imsave(final_path + "test/ground_truth/images/" + str(i) + ".png", ground_truth[i], cmap=cmap)

# MAIN

mnist = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.Compose([
                    transforms.Resize((size, size)),            
                    transforms.ToTensor()
                ]))

mnist_noise = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.Compose([
                    transforms.Resize((size, size)),            
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    GaussianNoiseTransform(0.0, 2.0)
                ]))  

#plt.imshow(mnist_noise[0][0][0], cmap = "gray")
#plt.show()

data = []
ground_truth = []
for i in range(dataset_size):
    data.append(mnist_noise[i][0][0].numpy())
    ground_truth.append(mnist[i][0][0].numpy())

# generating dataset with 15000 data 80-20 split (12000 training 3000 test)
generate_raw_dataset_split(data, ground_truth, cmap=cmap, path="./Dataset/"+str(size)+"x"+str(size), dataset_name="Dataset15000")