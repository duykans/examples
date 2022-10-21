import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#Image Save
import matplotlib
import os

# Dataset Parameters
dataset_path = "./Dataset/64x64/Dataset15000"
test_data_path = dataset_path + "/test/data" 
test_truth_path = dataset_path + "/test/ground_truth" 
imgsize_x = 64
imgsize_y = 64

#Testing parameters
batch_size = 1
val_epoch = 1

#Display and save parameters
val_interval = 9
output_display_threshold = False
threshold_vals = [0,1]
image_save = True
#Display, load and save paths
display = False 
model_path = "./ModelsToCompare/MNIST"
model_name = "/MnistUnetTL_Lambda001/model_unet_tl_lambda001_50ep.pt"
save_path = "./Results/UNetTL_Lambda001"

#Method used to save images
def imgsave(image, ground_truth, path, index):
    #Checking if the parh exists
    if not os.path.exists(path):
        os.makedirs(path + "/results") 
        os.makedirs(path + "/ground_truth") 
    #Saving Result image 
    matplotlib.image.imsave(path + "/results/" + str(index) +".png", image, cmap="gray")
    matplotlib.image.imsave(path + "/ground_truth/" + str(index) +".png", ground_truth, cmap="gray")

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

# U-Net
# Detailed Information: https://arxiv.org/pdf/1505.04597.pdf
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


#MAIN
#Dataset trasformation
transform = transforms.Compose(
   [transforms.Grayscale(num_output_channels=1),
    transforms.Resize((imgsize_x, imgsize_y)),
     transforms.ToTensor(),
     transforms.Normalize([0.0], [1.0]) #note: this does not normalise
    ])
    #transforms.Normalize([0.6], [0.6])

#Reading Dataset
test_data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_data_path, transform=transform),
    batch_size=batch_size,
    num_workers=0
    )

test_truth_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_truth_path, transform=transform),
    batch_size=batch_size,
    num_workers=0
    )


#Creaing Model
model = UNet(1).to(torch.device("cuda")) 

#Loading Weights
model = torch.load(model_path + model_name)

criterion = torch.nn.BCEWithLogitsLoss()

for i in range(val_epoch):
    total_test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(zip(test_data_loader, test_truth_loader)):
            target = data[1][0].cuda()
            data = data[0][0].cuda()
            
            #Feeding data to already trained network
            output = model(data)
            
            #Saving result image
            if image_save == True:
                imgsave(output[0][0].cpu().detach().numpy(), target[0][0].cpu().detach().numpy(), save_path, i)
            
            #Calculating loss with test data
            test_loss = criterion(output, target)
            total_test_loss = total_test_loss + test_loss
            
            #displaying
            if batch_idx % val_interval == 0:
                print("Test epoch : test loss current avarage : ", total_test_loss / (batch_idx + 1),
                        " Current Loss :", test_loss)
                #Showing resulting images
                if display: 
                    fig, ((ax1, ax2, ax3))= plt.subplots(1, 3, figsize=(8, 6))
                    fig.suptitle('Sharing x per column, y per row')
                    ax1.imshow(data[0][0].cpu().detach().numpy(), cmap="gray")
                    if output_display_threshold:
                        ax2.imshow(output[0][0].cpu().detach().numpy(), cmap="gray", vmin = threshold_vals[0], vmax = threshold_vals[1])
                    else :
                        ax2.imshow(output[0][0].cpu().detach().numpy(), cmap="gray")
                    ax3.imshow(target[0][0].cpu().detach().numpy(), cmap="gray")
                    plt.show()