from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
import foolbox
from foolbox.criteria import Misclassification
from foolbox.criteria import TargetClassProbability
from foolbox.distances import MSE

torch.manual_seed(7)

pretrained_model = '/home/shu30/research/paper/model/net_model1.pkl'
use_cuda=True

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/shu30/Downloads/data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/shu30/Downloads/data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),batch_size=10, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

dataiter = iter(train_loader)
img, label = dataiter.next()
img,label = img.to(device), label.to(device)

image = img[2]
label1 = label[2]

img_numpy = image.cpu().numpy().reshape(1,1,28,28)
label_numpy = label1.cpu().numpy().reshape(1)

# transfrom the model to the pytorch form
attacked_model1 = foolbox.models.PyTorchModel(model, bounds=(0,1), num_classes=10)


for j in np.array((0,2,3,4,5,6,7,8,9)):
#target attack
    ADef_attack_target = foolbox.attacks.ADefAttack(attacked_model1,criterion = TargetClassProbability(j,0.35))
# generate adversarial sample from PepperAndSaltNoiseAttack and then pass it to Pointwise attack
    adver_img1_target = np.zeros(shape=(100, 28, 28))
    label_ADef_target = np.zeros(100)
    for i in np.arange(0, 100, 1):  #100 here tries to match the 
    # Try the target model
    # define the SaltAndPepperNoise_attack
        pred_label = 1
        while pred_label != j:
    # Generate the adversarial samples with different choice of smooth parameter
            adversarial_images_target_ADef = ADef_attack_target(
            img_numpy.reshape(1, 1, 28, 28), label_numpy, smooth = 0.015 * i)
            adversarial_target_ADef = adversarial_images_target_ADef.reshape(
            28, 28)
            adversarial_image_target_ADef = torch.from_numpy(
            adversarial_target_ADef)

    # predict
            pred_label = np.argmax(
            attacked_model1.forward(
                adversarial_images_target_ADef))
        
        
        label_ADef_target[i] = pred_label
        adver_img1_target[i] = adversarial_image_target_ADef

    np.savetxt("./ADef/label_ADef_tar"+str(j),label_ADef_target)
    np.savetxt("./ADef/image_ADef_tar"+str(j),
           adver_img1_target.reshape(100 * 28 * 28))
