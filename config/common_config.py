"""
config of data path and model training
"""

import sys
sys.path.append("../")

clean_data_dir = '../clean_dataset' # defaul clean dataset directory
extra_data_dir = "../extra_val_set" # clean extra clean validation dataset directory

# arch setting
from models import resnet, vgg

arch = {
    ### base model of training
    'cifar10': resnet.ResNet18,
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
