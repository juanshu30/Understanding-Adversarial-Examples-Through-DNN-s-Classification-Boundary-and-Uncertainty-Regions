from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import argparse
from torch.autograd import Variable
from collections import Counter
import copy
import foolbox
from foolbox.criteria import TargetClassProbability
import pickle

# Define what device we are using
use_cuda = True
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# The Lenet model structure used
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


model = Net()

#Load Model1 to Model10

generate_model = locals()

for i in range(1, 11):
    name = 'pretrained_model' + str(i)
    generate_model['pretrained_model' + str(
        i)] = "./model/net_model" + str(i) + ".pkl"
    generate_model["model" + str(i)] = Net().to(device)


pretrained_model_list = [pretrained_model1, pretrained_model2, pretrained_model3, pretrained_model4,
                         pretrained_model5, pretrained_model6, pretrained_model7, pretrained_model8, 
                         pretrained_model9, pretrained_model10]
model_list = [
    model1, model2, model3, model4, model5, model6, model7, model8, model9,
    model10
]

generate_attacked_model = locals()
for i in np.arange(0,10):
    model_list[i].load_state_dict(torch.load(pretrained_model_list[i], map_location='cpu'))
    model_list[i].eval()
    name = 'attacked_model' + str(i+1)
    generate_attacked_model['attacked_model' + str(i+1)] = foolbox.models.PyTorchModel(
        model_list[i], bounds=(0, 1), num_classes=10)
    
    
# Load the original image "1"
original_image1 = np.loadtxt(
    "./original_image/original_image0.txt").reshape(28, 28)

# Load the adversarial images
generate_image_var = locals()
for i in np.array((1, 2, 3, 4, 5, 6, 7, 8, 9)):
    image_var = "image1_"+str(i)
    generate_image_var["image1_"+str(i)] = np.loadtxt(
        "./BIML1/image_BIML1_tar" + str(i)).reshape(100, 28, 28)
    
    
    
def IntervalMethods(image_1, adver_image, attackedModel):
    # the input of the function are the adversarial images, original image,

    # calculate the difference between original image and the adversarial images
    diff = np.zeros(shape=(len(adver_image), 784))
    for i in np.arange(0, len(adver_image), 1):
        diff[i] = np.round(
            (adver_image[i].reshape(784) - image_1.reshape(784)),
            3)

    position = list()
    for i in np.arange(0, len(diff), 1):
        position.append(np.where(diff[i] != 0))

    position_used = np.concatenate(np.concatenate(np.array(position)))
    table_change_position = Counter(position_used)
    position_freq = np.array(table_change_position.most_common())
    position_freq100 = position_freq[
        position_freq[:, 1] == 100,
    ]  # change 100 here if you don't want the pixels
    # that changed 100 times in the adver_image

    pixels_freq100 = position_freq100[:, 0]
    pixels_length = len(pixels_freq100)

    adver_image_784 = adver_image.reshape(100, 784)

    # Find the key pixels

    def my_func_dim(x, d):
        return x[d]

    min_max = np.zeros(shape=(pixels_length, 2))
    for i in np.arange(0, pixels_length, 1):
        min_value = min(
            np.apply_along_axis(my_func_dim, 1, adver_image_784,
                                pixels_freq100[i]))
        max_value = max(
            np.apply_along_axis(my_func_dim, 1, adver_image_784,
                                pixels_freq100[i]))
        min_max[i] = (min_value, max_value)

    interval_diff = np.zeros(len(min_max))
    for i in np.arange(0, len(min_max)):
        interval_diff[i] = min_max[i][1] - min_max[i][0]

    # stack all the key dimensions, min_max and the intervals together
    interval_table = np.c_[min_max, interval_diff, pixels_freq100]

    # sort the above idimensions by the len of intervals(from least to the largest)
    interval_table_sorted = np.array(
        sorted(interval_table, key=lambda x: (-x[2])))

    main_dim_sorted = interval_table_sorted[:, 3]

    # Test whether each model can missclassify the original adversarial images
    pred_ori_adver = np.zeros(len(adver_image))
    for k in np.arange(0, len(adver_image)):
        pred_ori_adver[k] = np.argmax(
            attackedModel.forward(adver_image[k].reshape(
                1, 1, 28, 28).astype(np.float32)))
    miss_rate_ori_adver = sum(pred_ori_adver != 0) / len(adver_image)

    # generate new adversarial images based on the entire interval
    p = 0
    pred_neiLR = np.zeros(20)
    for k in np.arange(pixels_length - 200,
                       pixels_length, 10):
        adver_add_ori = np.zeros(shape=(500, 28, 28))
        for j in np.arange(0, 500, 1):
            a_image = copy.deepcopy(original_image1).reshape(784)
            for i in np.arange(0, len(main_dim_sorted[0:k]), 1):
                a_image[int(main_dim_sorted[i])] = np.random.uniform(interval_table_sorted[:, 0][i],
                                                                     interval_table_sorted[:, 1][i])
            adver_add_ori[j] = a_image.reshape(28, 28)

        predict_whole = np.zeros(len(adver_add_ori))
        for i in np.arange(0, len(adver_add_ori), 1):
            predict_whole[i] = np.argmax(attacked_model1.forward(
                adver_add_ori[i].reshape(1, 1, 28, 28).astype(np.float32)))

        pred_neiLR[p] = sum(predict_whole != 0) / len(predict_whole)
        p += 1


#       generate new adversarial images based on the (min-0.05, max+0.05)

    p = 0
    pred_neiLR_01 = np.zeros(10)
    for k in np.arange(30,90,10):
        adver_add_ori = np.zeros(shape=(500, 28, 28))
        for j in np.arange(0, 500, 1):
            a_image = copy.deepcopy(original_image1).reshape(784)
            for i in np.arange(0, len(main_dim_sorted[0:k]), 1):
                a_image[int(main_dim_sorted[i])] = np.random.uniform(0,1)
            adver_add_ori[j] = a_image.reshape(28, 28)

        predict_whole = np.zeros(len(adver_add_ori))
        for i in np.arange(0, len(adver_add_ori), 1):
            predict_whole[i] = np.argmax(attacked_model1.forward(
                adver_add_ori[i].reshape(1, 1, 28, 28).astype(np.float32)))

        pred_neiLR_01[p] = sum(predict_whole != 0) / len(predict_whole)
        p += 1

        # generate new adversarial images based on the (original-0.1, original+0.1)

    return (pixels_freq100, pixels_length, interval_table_sorted, main_dim_sorted, miss_rate_ori_adver,pred_neiLR, pred_neiLR_01)




image_list = (image1_1, image1_2, image1_3, image1_4,
              image1_5, image1_6, image1_7, image1_8, image1_9)
attack_model_list = (attacked_model1, attacked_model2, attacked_model3, attacked_model4, attacked_model5,
                     attacked_model6, attacked_model7, attacked_model8, attacked_model9,attacked_model10)


results_all = list()
for i in np.arange(0,10):
    for j in np.arange(0,9):
         results_all.append(IntervalMethods(original_image1,image_list[j],attack_model_list[i]))
            
            
import pickle
file=open(r"./BIML1/results_BIML1.bin","wb")
pickle.dump(results_all,file)
file.close()
