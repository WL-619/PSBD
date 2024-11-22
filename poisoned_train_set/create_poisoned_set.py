'''
Modified from https://github.com/vtu81/backdoor-toolbox/blob/main/create_poisoned_set.py
'''
import os
import sys
sys.path.append("..")
import torch
from torchvision import datasets, transforms
from dataset.GTSRB import GTSRB
import argparse
from PIL import Image
import numpy as np
from config import attack_config, common_config, default_args
from utils import supervisor, tools


def add_arguments(parser):
    """
        Args:
            dataset (str): The dataset we used
            
            poison_type (str): Attack types

            poisoning_ratio (float): Proportion of backdoor data in the entire training dataset

            cover_rate (float): Ratio of data samples with backdoor trigger but without label modification to target class

            alpha (float): Blend rate for blend or adaptive_blend attacks

            trigger (str): trigger of attacks

            poison_seed (int): Random seed for generating the poisoned dataset, rather than the seed used for model training

    """

    parser.add_argument('-dataset', type=str, required=False,
                        default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'])
    parser.add_argument('-poison_type', type=str,  required=False,
                        choices=default_args.parser_choices['poison_type'],
                        default=default_args.parser_default['poison_type'])
    parser.add_argument('-poisoning_ratio', type=float,  required=False,
                        choices=default_args.parser_choices['poisoning_ratio'],
                        default=default_args.parser_default['poisoning_ratio'])
    parser.add_argument('-cover_rate', type=float,  required=False,
                        choices=default_args.parser_choices['cover_rate'],
                        default=default_args.parser_default['cover_rate'])
    parser.add_argument('-alpha', type=float,  required=False,
                        default=default_args.parser_default['alpha'])
    parser.add_argument('-trigger', type=str,  required=False,
                        default=None)
    parser.add_argument('-poison_seed', type=int, default=attack_config.poison_seed)
    return parser

def prepare_dataset(args):
    attack_config.record_poison_seed = True
    clean_data_dir = common_config.clean_data_dir
    if args.trigger is None:
        args.trigger = attack_config.trigger_default[args.dataset][args.poison_type]

    if not os.path.exists(os.path.join('../poisoned_train_set', args.dataset)):
        os.mkdir(os.path.join('../poisoned_train_set', args.dataset))


    img_size, num_classes = supervisor.get_info_dataset(args)


    if args.dataset == 'cifar10':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(
            os.path.join(clean_data_dir, 'cifar10'), 
            train=True,
            download=True, 
            transform=data_transform
        )
    elif args.dataset == 'gtsrb':
        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = GTSRB(
            os.path.join(clean_data_dir, 'gtsrb'),
            train=True,
            transform=data_transform
        )
    elif args.dataset == 'tiny':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(
            os.path.join(os.path.join(clean_data_dir, 'tiny'), 'train'),
            data_transform
        )

    else:
        raise  NotImplementedError('Dataset is not implemented')

    trigger_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    poison_set_save_dir = supervisor.get_poison_set_dir(args)

    if os.path.exists(poison_set_save_dir):
        print(f"Poisoned set directory '{poison_set_save_dir}' to be created is not empty! Will override the directory!")
    else:
        tools.create_missing_folders(poison_set_save_dir)

    trigger_name = args.trigger
    trigger_path = os.path.join(attack_config.triggers_dir, trigger_name)

    trigger = None
    trigger_mask = None

    if trigger_name != 'none':

        print('trigger: %s' % trigger_path)
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        trigger_mask_path = os.path.join(attack_config.triggers_dir, 'mask_%s' % trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

    alpha = args.alpha

    poison_generator = None
        
    if args.poison_type == 'badnet':

        from attack import badnet
        poison_generator = badnet.poison_generator(
            img_size=img_size, 
            dataset=train_set,
            poisoning_ratio=args.poisoning_ratio, 
            trigger_mark=trigger, 
            trigger_mask=trigger_mask,
            target_label=attack_config.target_label[args.dataset]
        )

    elif args.poison_type == 'blend':

        from attack import blend
        poison_generator = blend.poison_generator(
            img_size=img_size, dataset=train_set,
            poisoning_ratio=args.poisoning_ratio, trigger=trigger,
            target_label=attack_config.target_label[args.dataset],
            alpha=alpha
        )

    elif args.poison_type == 'wanet':
        # prepare grid
        s = 0.5
        k = 4
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            torch.nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=img_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        
        path = os.path.join(poison_set_save_dir, 'identity_grid.pt')
        torch.save(identity_grid, path)
        path = os.path.join(poison_set_save_dir, 'noise_grid.pt')
        torch.save(noise_grid, path)

        from attack import wanet
        poison_generator = wanet.poison_generator(
            img_size=img_size, 
            dataset=train_set,
            poisoning_ratio=args.poisoning_ratio, 
            cover_rate=args.cover_rate,
            identity_grid=identity_grid, 
            noise_grid=noise_grid,
            s=s, 
            k=k, 
            grid_rescale=grid_rescale, 
            target_label=attack_config.target_label[args.dataset]
        )

    elif args.poison_type == 'adaptive_blend':
        from attack import adaptive_blend

        poison_generator = adaptive_blend.poison_generator(
            img_size=img_size, 
            dataset=train_set,
            poisoning_ratio=args.poisoning_ratio,
            cover_rate=args.cover_rate,
            trigger=trigger,
            target_label=attack_config.target_label[args.dataset],
            pieces=16, 
            mask_rate=0.5,
            alpha=alpha,
        )

    elif args.poison_type == 'lc':

        if args.dataset == 'cifar10':
            adv_imgs_path = "./cifar10/lc/fully_poisoned_training_datasets/two_600.npy"
            if not os.path.exists(adv_imgs_path):
                raise NotImplementedError("Run './cifar10/lc_cifar10.sh' first to launch clean label attack!")
            adv_imgs_src = np.load(adv_imgs_path).astype(np.uint8)
            adv_imgs = []
            for i in range(adv_imgs_src.shape[0]):
                adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
            adv_imgs = torch.cat(adv_imgs, dim=0)
            assert adv_imgs.shape[0] == len(train_set)
        else:
            raise NotImplementedError('Label-Consistent Attack is not implemented for %s' % args.dataset)

        # init attacker
        from attack import lc
        poison_generator = lc.poison_generator(
            img_size=img_size, 
            dataset=train_set, 
            adv_imgs=adv_imgs,
            poisoning_ratio=args.poisoning_ratio,
            trigger_mark = trigger, trigger_mask=trigger_mask,
            path=poison_set_save_dir,
            target_label=attack_config.target_label[args.dataset],
        )


    if args.poison_type not in ['wanet', 'adaptive_blend']:
        clean_img_set, clean_label_set, backdoor_img_set, backdoor_label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % (len(clean_label_set)+len(backdoor_label_set)))

    else:
        clean_img_set, clean_label_set, backdoor_img_set, backdoor_label_set, cover_img_set, cover_label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % (len(clean_label_set)+len(backdoor_label_set)+len(cover_label_set)))

        cover_img_path = os.path.join(poison_set_save_dir, 'cover_imgs.pt')
        torch.save(cover_img_set, cover_img_path)
        print('[Generate Poisoned Set] Save %s' % cover_img_path)

        cover_label_path = os.path.join(poison_set_save_dir, 'cover_labels.pt')
        torch.save(cover_label_set, cover_label_path)
        print('[Generate Poisoned Set] Save cover label %s' % cover_label_path)


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

    print('[target class : %d]' % attack_config.target_label[args.dataset])

    prepare_dataset(args)