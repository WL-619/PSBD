'''
Modified from https://github.com/vtu81/backdoor-toolbox/blob/main/create_clean_set.py
'''
import sys
sys.path.append("..")

from torchvision import transforms
import os
import torch
from torchvision import datasets
import argparse
import random
from config import common_config, default_args
from utils import tools, supervisor

def add_arguments(parser):
    """
        Args:
            dataset (str): The dataset we used

            random_seed (int): Random seed for generating the clean extra validation dataset, rather than the seed used for model training

            val_budget_rate (float, default=0.05): Proportion of clean extra validation data compared to the entire poisoned training set

    """
    parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'])
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-val_budget_rate', type=float, default=0.05)
    return parser

def prepare_dataset(args):
    data_dir = common_config.clean_data_dir  # directory to save standard clean set
    data_transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'cifar10':
        clean_set = datasets.CIFAR10(
            root=os.path.join(data_dir, 'cifar10'), 
            train=False,
            download=True,
            transform=data_transform,
        )
    elif args.dataset == 'tiny':
        clean_set = datasets.ImageFolder(
            root=os.path.join(data_dir, 'tiny', 'val'),
            transform=data_transform,
        )
    else:
        raise NotImplementedError('Dataset is not implemented')

    # extra validation data: clean samples at hand for defensive purpose
    extra_val_split_dir = os.path.join(args.dataset, 'extra_val_split', 'val_budget_rate_'+str(args.val_budget_rate))
    if not os.path.exists(extra_val_split_dir): tools.create_missing_folders(extra_val_split_dir)

    test_split_dir = os.path.join(args.dataset, 'test_split')  # test samples for evaluation purpose
    if not os.path.exists(test_split_dir): tools.create_missing_folders(test_split_dir)

    if args.dataset in default_args.parser_choices['dataset']:

        # randomly sample from a clean test/val set to simulate the extra clean val samples at hand
        if(args.dataset != 'tiny'):
            num_train_imgs, num_test_imgs = supervisor.get_info_dataset(args, get_count=True)
        else:
            num_train_imgs, num_val_imgs, num_test_imgs = supervisor.get_info_dataset(args, get_count=True)

        num_img = num_train_imgs
        id_set = list(range(0, len(clean_set)))
        random.shuffle(id_set)
        extra_val_split_indices = id_set[:int(num_img * args.val_budget_rate)]
        test_indices = id_set[int(num_img * args.val_budget_rate):]

        extra_val_split_set = torch.utils.data.Subset(clean_set, extra_val_split_indices)
        torch.save(extra_val_split_set, os.path.join(extra_val_split_dir, 'extra_val_data.pt'))
        print('[Generate Clean Extra Validation Split Set] Save %s' % os.path.join(extra_val_split_dir, 'extra_val_data.pt'))

        test_split_set = torch.utils.data.Subset(clean_set, test_indices)
        torch.save(test_split_set, os.path.join(test_split_dir, 'test_data.pt'))
        print('[Generate Clean Test Split Set] Save %s' % os.path.join(test_split_dir, 'test_data.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    tools.setup_seed(args.random_seed)
    prepare_dataset(args)