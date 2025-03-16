"""
Configuration file for data path and model training settings
"""

import sys
sys.path.append("../")

clean_data_dir = '../clean_dataset' # Defaul clean dataset directory
extra_data_dir = "../extra_val_set" # Clean extra validation dataset directory

# Model architecture settings
from models import resnet, vgg

arch = {
    # Base model architecture of training
    'cifar10': resnet.ResNet18,
    'gtsrb': resnet.ResNet18,
    'tiny': resnet.ResNet18,
    # 'cifar10': vgg.vgg16_bn,
    # 'tiny': vgg.vgg16_bn,
}

record_model_arch = True

training_setting = {
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'epochs': 100,
    'milestones': [50, 75],
    'learning_rate': 0.1,
    'batch_size': 128,
}
