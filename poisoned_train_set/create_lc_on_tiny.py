import os
import sys
sys.path.append("../")
import torch
from torchvision import datasets, transforms
import numpy as np
from config import common_config, attack_config
from utils import supervisor, tools
import argparse
from PIL import Image
import random

def add_arguments(parser):
    """
        Args:
            poisoning_ratio (float): Proportion of backdoor data in the entire training dataset

            adv_data_path (str): Path of lc data on the tiny imagenet

            alpha (float): Blend rate for blend or adaptive_blend attacks

            poison_seed (int): Random seed for generating the poisoned dataset, rather than the seed used for model training
    """

    parser.add_argument('-dataset', type=str, default='tiny')
    parser.add_argument('-poison_type', type=str, default='lc')
    parser.add_argument('-poisoning_ratio', type=float)
    parser.add_argument('-adv_data_path', type=str)
    parser.add_argument('-alpha', type=float, default=0.05)
    parser.add_argument('-poison_seed', type=int, default=attack_config.poison_seed)
    return parser

def prepare_dataset(args):
    target_label = attack_config.target_label[args.dataset]
    print('[target label : %d]' % attack_config.target_label[args.dataset])
    attack_config.record_poison_seed = True

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    img_size, num_classes = supervisor.get_info_dataset(args)
    trigger_name = 'badnet_patch4_64.png'
    trigger_path = os.path.join(attack_config.triggers_dir, trigger_name)

    trigger = Image.open(trigger_path).convert("RGB")
    trigger = transform(trigger)

    trigger_mask_path = os.path.join(attack_config.triggers_dir, 'mask_%s' % trigger_name)
    trigger_mask = Image.open(trigger_mask_path).convert("RGB")
    trigger_mask = transform(trigger_mask)[0]

    from attack import badnet
    poison_transform = badnet.poison_transform(img_size, trigger, trigger_mask, target_label, args.alpha)

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
    elif args.dataset == 'tiny':
        clean_train_data = datasets.ImageFolder(
            root=clean_data_dir+'/train'
        )
    else:
        raise NotImplementedError('Dataset is not implemented')

    adv_data_dir = args.adv_data_path
    adv_train_set = np.load(adv_data_dir)

    id_set = list(range(0, len(adv_train_set)))
    random.shuffle(id_set)
    num_poison = int(len(adv_train_set) * args.poisoning_ratio)
    backdoor_indices = id_set[:num_poison]
    clean_indices = id_set[num_poison:]

    backdoor_indices.sort() # increasing order
    clean_indices.sort()

    print('number of backdoor samples: ', len(backdoor_indices))

    clean_img_set = []
    backdoor_img_set = []

    clean_label_set = []
    backdoor_label_set = []


    for i in backdoor_indices:
        img = transform(adv_train_set[i])
        img = poison_transform.transform(img, torch.tensor([0]))[0]
        
        backdoor_img_set.append(img)
        backdoor_label_set.append(target_label)

    for i in clean_indices:
        img, gt = clean_train_data[i]
        clean_img_set.append(transform(img))
        clean_label_set.append(gt)

    clean_label_set = torch.LongTensor(clean_label_set)
    backdoor_label_set = torch.LongTensor(backdoor_label_set)

    clean_img_path = os.path.join(poison_set_save_dir, 'clean_imgs.pt')
    torch.save(clean_img_set, clean_img_path)
    print('[Generate Poisoned Set] Save clean imgs %s' % clean_img_path)

    backdoor_img_path = os.path.join(poison_set_save_dir, 'backdoor_imgs.pt')
    torch.save(backdoor_img_set, backdoor_img_path)
    print('[Generate Poisoned Set] Save backdoor imgs %s' % backdoor_img_path)

    clean_label_path = os.path.join(poison_set_save_dir, 'clean_labels.pt')
    torch.save(clean_label_set, clean_label_path)
    print('[Generate Poisoned Set] Save clean label %s' % clean_label_path)

    backdoor_label_path = os.path.join(poison_set_save_dir, 'backdoor_labels.pt')
    torch.save(backdoor_label_set, backdoor_label_path)
    print('[Generate Poisoned Set] Save backdoor label %s' % backdoor_label_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    tools.setup_seed(args.poison_seed)
    prepare_dataset(args)