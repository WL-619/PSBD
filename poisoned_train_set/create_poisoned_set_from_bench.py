import os
import sys
sys.path.append("../")
import torch
from torchvision import datasets, transforms
import numpy as np
from config import common_config
from utils import supervisor, tools
from dataset.GTSRB import GTSRB
import argparse

def add_arguments(parser):
    """
        Args:
            dataset (str): The training dataset
            
            poison_type (str): Attack type

            data_path (str): Path of bench data

            poisoning_ratio (float): Proportion of backdoor data in the entire training dataset
            
    """

    parser.add_argument('-dataset', type=str)
    parser.add_argument('-poison_type', type=str)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-poisoning_ratio', type=float)
    return parser

def prepare_dataset(args):
    poison_set_save_dir = supervisor.get_poison_set_dir(args)

    if os.path.exists(poison_set_save_dir):
        print(f"Poisoned set directory '{poison_set_save_dir}' to be created is not empty! Will override the directory!")
    else:
        tools.create_missing_folders(poison_set_save_dir)

    clean_data_dir = common_config.clean_data_dir
    clean_data_dir = os.path.join(clean_data_dir, args.dataset)

    if args.dataset == 'cifar10':
        clean_train_data = datasets.CIFAR10(
            root=clean_data_dir, 
            train=True
        )
    elif args.dataset == 'gtsrb':
        clean_train_data = GTSRB(
            clean_data_dir,
            train=True
        )
    elif args.dataset == 'tiny':
        clean_train_data = datasets.ImageFolder(
            root=clean_data_dir+'/train'
        )
    else:
        raise NotImplementedError('Dataset is not implemented')


    load_path = os.path.join(args.data_path, 'attack_result.pt')
    load_file = torch.load(load_path)

    poison_indicator = load_file['bd_train']['poison_indicator']
    poison_indicator_path = os.path.join(poison_set_save_dir, 'poison_indicator.pt')
    torch.save(torch.tensor(poison_indicator), poison_indicator_path)
    print('[Generate Poisoned Indicator] Save poison_indicator %s' % poison_indicator_path)


    posion_index = np.where(poison_indicator == 1)[0]
    clean_index = np.where(poison_indicator == 0)[0]


    # np.savetxt(poison_set_save_dir+'/posion_index.txt', posion_index, fmt='%d')
    # np.savetxt(poison_set_save_dir+'/clean_index.txt', clean_index, fmt='%d')

    def dataset_transform(dataset):
        transformed_imgs = []
        transformed_labels = []
        for img, label in dataset:
            transformed_imgs.append(img)
            transformed_labels.append(label)

        transformed_imgs = torch.stack(transformed_imgs, dim=0)
        transformed_labels = torch.tensor(transformed_labels)
        return torch.utils.data.TensorDataset(transformed_imgs, transformed_labels)

    clean_data = torch.utils.data.Subset(clean_train_data, clean_index)
    backdoor_data = datasets.ImageFolder(
        root=os.path.join(args.data_path, 'bd_train_dataset'), 
        transform=transforms.ToTensor()
    )
    backdoor_data = dataset_transform(backdoor_data)

    backdoor_test_data = datasets.ImageFolder(
        root=os.path.join(args.data_path, 'bd_test_dataset'), 
        transform=transforms.ToTensor()
    )
    backdoor_test_data = dataset_transform(backdoor_test_data)


    clean_data_path = os.path.join(poison_set_save_dir, 'clean_train_data.pt')
    torch.save(clean_data, clean_data_path)
    print('[Generate Clean Train Set] Save clean_data %s' % clean_data_path)

    backdoor_data_path = os.path.join(poison_set_save_dir, 'backdoor_train_data.pt')
    torch.save(backdoor_data, backdoor_data_path)
    print('[Generate Backdoor Set] Save backdoor_data %s' % backdoor_data_path)

    backdoor_test_path = os.path.join(poison_set_save_dir, 'backdoor_test_data.pt')
    torch.save(backdoor_test_data, backdoor_test_path)
    print('[Generate Backdoor Test Set] Save backdoor_test_data %s' % backdoor_test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    prepare_dataset(args)