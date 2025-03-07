import os
import sys
import argparse

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets

from utils import generate_concepts

sys.path.append("..")
from lib.train import Trainer
from lib.models import ConvNet

### Converts grayscale image to either red or green
def color_grayscale_arr(arr, red=True):

  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = torch.reshape(arr, [h, w, 1])
  if red:
    arr = torch.cat((arr,
                    torch.zeros((h, w, 2), dtype=dtype)), axis=2)
  else:
    arr = torch.cat((torch.zeros((h, w, 1), dtype=dtype),
                     arr,
                     torch.zeros((h, w, 1), dtype=dtype)), axis=2)
  return arr


### assigns color labels according to color label
def assign_color_label(data, color_label, binary_label):

    for idx in range(len(binary_label)):
        sample_data = data[idx]
        sample_color_label = color_label[idx]

        sample_data = color_grayscale_arr(sample_data, red=(sample_color_label == 1)) + 0.0
        sample_data /= 255.0
        sample_data = sample_data.unsqueeze(0)

        if idx == 0:
            data_with_shortcut = sample_data
        else:
            data_with_shortcut = torch.cat((data_with_shortcut, sample_data), dim=0)

        if (idx+1)%500 == 0:
            print(f"{idx+1}/{len(binary_label)} samples processed.")

    print(f"Data shape: {data_with_shortcut.shape}")
    
    return data_with_shortcut


class ColoredMnist():

    def __init__(self, dataset_name, random_state=42):

        self.dataset = datasets.MNIST("./colored_mnist", train=False, download=True)
        self.random_state = random_state

    '''
    Generate train_idx, train_binary_label, train_color_label, 
             test_idx, test_binary_label, test_color_label
    '''
    def get_train_test_idx(self, shortcut_ratio):

        self.shortcut_ratio = shortcut_ratio

        img_data, img_label = self.dataset.data, self.dataset.targets
        
        np.random.seed(self.random_state)
        self.sample_num = len(img_label)
        self.all_idx = list(range(self.sample_num))
        self.all_binary_label = (img_label >= 5) + 0.0

        ### randomly select the indices of the training and test dataset
        self.train_idx = np.random.choice(self.all_idx, size=int(self.sample_num/2), replace=False)
        ## binarize the labels
        self.train_binary_label = self.all_binary_label[self.train_idx]

        ### assign color labels the same as the binary label
        self.train_color_label = self.train_binary_label[0:int(self.sample_num/2*self.shortcut_ratio)]
        self.train_color_label = np.concatenate((self.train_color_label, np.random.binomial(1, 0.5, int(self.sample_num/2*(1-self.shortcut_ratio)))))
        self.train_color_label = torch.from_numpy(self.train_color_label) + 0.0

        self.test_idx = list(set(self.all_idx) - set(self.train_idx))
        self.test_binary_label = self.all_binary_label[self.test_idx]

        ## correlation between shortcut and label is reversed in test dataset
        self.test_color_label = (img_label[self.test_idx] < 5) + 0.0

        self.train_data = img_data[self.train_idx]
        self.test_data = img_data[self.test_idx]
    

    def assign_shortcuts(self):
        
        self.train_data = assign_color_label(self.train_data, self.train_color_label, self.train_binary_label)
        self.test_data = assign_color_label(self.test_data, self.test_color_label, self.test_binary_label)

        ### permute to batcu_size * in_channel * height * weight
        self.train_data = self.train_data.permute(0,3,1,2)
        self.test_data = self.test_data.permute(0,3,1,2)

        return self.train_data, self.train_binary_label, self.train_color_label, self.test_data, self.test_binary_label, self.test_color_label
    

    ### generate unbiased training data for concepts
    def generate_unbiased_data(self):
        
        np.random.seed(self.random_state)

        self.unbiased_color_label = np.random.binomial(1, 0.5, self.train_data.shape[0])
        self.unbiased_color_label = torch.from_numpy(self.unbiased_color_label) + 0.0
        self.unbiased_train_data = assign_color_label(self.train_data, self.unbiased_color_label, self.train_binary_label)
        self.unbiased_train_data = self.unbiased_train_data.permute(0,3,1,2)

        return self.unbiased_train_data, self.unbiased_color_label


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Colored MNIST dataset generation and shortcut features assigning...')
    # parser.add_argument('--shortcut_ratio', type=float, default=1.0,
    #                     help='Ratio of shortcuts in training data (default: 1.0)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for data generation (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run training on (default: cpu)')
    args = parser.parse_args()

    path = "./data/colored_mnist"
    # Create colored_mnist directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created ./colored_mnist directory")

    # Check if data files exist
    required_files = [
        "train_data.pt", "train_binary_label.pt", "train_color_label.pt",
        "test_data.pt", "test_binary_label.pt", "test_color_label.pt",
        "unbiased_train_data.pt", "unbiased_color_label.pt"
    ]
    
    files_exist = all(os.path.exists(os.path.join(path, f)) for f in required_files)

    if not files_exist:
        print("Generating new data...")
        color_mnist = ColoredMnist("colored-mnist", random_state=args.random_state)
        color_mnist.get_train_test_idx(shortcut_ratio=1)
        unbiased_train_data, unbiased_color_label = color_mnist.generate_unbiased_data()
        train_data, train_binary_label, train_color_label, test_data, test_binary_label, test_color_label = color_mnist.assign_shortcuts()

        # Save generated data
        torch.save(train_data, f"{path}/train_data.pt")
        torch.save(train_binary_label, f"{path}/train_binary_label.pt")
        torch.save(train_color_label, f"{path}/train_color_label.pt")
        torch.save(test_data, f"{path}/test_data.pt")
        torch.save(test_binary_label, f"{path}/test_binary_label.pt")
        torch.save(test_color_label, f"{path}/test_color_label.pt")
        torch.save(unbiased_train_data, f"{path}/unbiased_train_data.pt")
        torch.save(unbiased_color_label, f"{path}/unbiased_color_label.pt")
        print(f"Saved data to {path}")
    else:
        print("Loading existing data...")

    # Load data
    train_data = torch.load(f"{path}/train_data.pt")
    unbiased_train_data = torch.load(f"{path}/unbiased_train_data.pt")
    train_label = torch.load(f"{path}/train_binary_label.pt")
    train_color_label = torch.load(f"{path}/train_color_label.pt")
    unbiased_color_label = torch.load(f"{path}/unbiased_color_label.pt")

    test_data = torch.load(f"{path}/test_data.pt")
    test_label = torch.load(f"{path}/test_binary_label.pt")
    test_color_label = torch.load(f"{path}/test_color_label.pt")

    loss_fn = nn.BCEWithLogitsLoss()
 
    print("Train on unbiased data for concepts extraction")
    cmnist_concept_model = Trainer(ConvNet(), device=args.device,
                                    train_data=unbiased_train_data, train_label=train_label, test_data=test_data, test_label=test_label,
                                    train_loss_fn=loss_fn, test_loss_fn=loss_fn)
    cmnist_concept_model.train()

    print("Train on unbiased data to predict color to extract shortcut features")
    cmnist_shortcut_model = Trainer(ConvNet(), device=args.device,
                                    train_data=unbiased_train_data, train_label=unbiased_color_label, test_data=test_data, test_label=test_color_label,
                                    train_loss_fn=loss_fn, test_loss_fn=loss_fn)
    cmnist_shortcut_model.train()


    cmnist_unknown_model = nn.Sequential(torchvision.models.resnet50(weights='DEFAULT'), nn.Linear(1000, 50))


    train_last_layer = generate_concepts(net_c=cmnist_concept_model.model,
                                         net_u=cmnist_unknown_model,
                                         net_s=cmnist_shortcut_model.model,
                                         data=train_data)
    
    test_last_layer = generate_concepts(net_c=cmnist_concept_model.model,
                                         net_u=cmnist_unknown_model,
                                         net_s=cmnist_shortcut_model.model,
                                         data=test_data)
    
    
    shortcut_ratio = args.shortcut_ratio
    torch.save(train_last_layer, f"{path}/train_last_layer.pt")
    torch.save(test_last_layer, f"{path}/test_last_layer.pt")