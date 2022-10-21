import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


#Parameters
dataset_path = "./Dataset/64x64/Dataset15000"
train_data_path = dataset_path + "/train/data" 
train_truth_path = dataset_path + "/train/ground_truth" 
#Training Parameters
batch_size = 1
lr = 0.005
epoch = 100
log_interval = 1000
save_interval = 10
epoch = 50

#Display, loading and saving parameters
display = False
load_model = False
model_path = "./Model/Unet"
model_name = "/model_unet"

def imshow(img):
    #img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Unet ,mplementation
#source: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

# U-Net
# Detailed information: https://arxiv.org/pdf/1505.04597.pdf
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

# MAIN

#Dataset trasformation
transform = transforms.Compose(
   [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize([0.0], [1.0]) #note: this does not normalise
    ])
    #transforms.Normalize([0.6], [0.6])

#Reading dataset
train_data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_data_path, transform=transform),
    batch_size=batch_size,
    num_workers=0
    )

train_truth_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_truth_path, transform=transform),
    batch_size=batch_size,
    num_workers=0
    )

#Creating Model model
model = UNet(1).to(torch.device("cuda")) 
summary(model, input_size=(1,64,64))

#seting up optimazer
optimizer = optim.SGD(model.parameters(), lr=lr)

#defining our loss function
criterion = torch.nn.BCEWithLogitsLoss()

#Loading Model
if load_model:
    print("loading model")
    model = torch.load(model_path + model_name)
    model.eval()

print("device : ", torch.cuda.current_device())
print("**********Training************")

#Training
ep = 0
model.train()
for i in range(epoch):
    ep = ep + 1
    for batch_idx, data in enumerate(zip(train_data_loader, train_truth_loader)):
        target = data[1][0].cuda()
        data = data[0][0].cuda()

        #data = data.cuda()
        #target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep,
                epoch, batch_idx * len(data), len(train_data_loader.dataset),
                100. * batch_idx / len(train_data_loader), loss.item()))
            if display: 
                fig, ((ax1, ax2, ax3))= plt.subplots(1,3)
                ax1.imshow(data[0][0].cpu().detach().numpy(), cmap="gray")
                ax2.imshow(output[0][0].cpu().detach().numpy(), cmap="gray")
                ax3.imshow(target[0][0].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
                plt.show()
    
    if ep % save_interval == 0:
        print("SAVING MODEL")
        torch.save(model, model_path + model_name + str(ep) + "ep.pt")