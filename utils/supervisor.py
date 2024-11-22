'''
Modified from https://github.com/vtu81/backdoor-toolbox/utils/supervisor.py
'''
import torch
import os
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("..")
from config import attack_config, common_config
from utils import tools

def get_model_dir(args, cleanse=False, defense=False):
    return args.model_path


def get_info_core(args, include_poison_seed=False):
    ratio = '%.3f' % args.poisoning_ratio
    if args.poison_type in ['trojannn']:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)
    elif args.poison_type == 'blend':
        blend_alpha = '%.3f' % args.alpha
        dir_core = '%s_%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'wanet':
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)

    if include_poison_seed:
        dir_core = f'{dir_core}_poison_seed={attack_config.poison_seed}'
    if common_config.record_model_arch:
        dir_core = f'{dir_core}_arch={get_arch(args).__name__}'
    return dir_core


def get_poison_set_dir(args, mkdir=False):
    ratio = '%.3f' % args.poisoning_ratio

    if args.poison_type == 'blend':
        blend_alpha = '%.3f' % args.alpha
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'wanet':
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        poison_set_dir = 'poisoned_train_set/%s/%s_%s' % (args.dataset, args.poison_type, ratio)

    poison_set_dir = f'../{poison_set_dir}_target_label={attack_config.target_label[args.dataset]}'

    if attack_config.record_poison_seed: 
        poison_set_dir = f'{poison_set_dir}_poison_seed={attack_config.poison_seed}'

    if mkdir:
        tools.create_missing_folders(poison_set_dir)
    return poison_set_dir


def get_arch(args):
    return common_config.arch[args.dataset]


def get_transforms(args):
    if args.dataset == 'cifar10':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

    elif args.dataset == 'gtsrb':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629],
                                     [1 / 0.2672, 1 / 0.2564, 1 / 0.2629])
            ])

    elif args.dataset == 'tiny':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])
            ])
    else:
        raise NotImplementedError('Dataset is not implemented')

    return data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer


def get_poison_transform(poison_type, dataset_name, target_label, trigger_transform=None, alpha=0.2, trigger_name=None, args=None):
    
    if trigger_name is None:
        trigger_name = attack_config.trigger_default[dataset_name][poison_type]

    if dataset_name in ['cifar10', 'gtsrb']:
        img_size = 32
    elif dataset_name == 'tiny':
        img_size = 64
    else:
        raise NotImplementedError('Dataset %s is not implemented' % dataset_name)

    if dataset_name == 'cifar10':
        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                 [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])
    elif dataset_name == 'gtsrb':
        normalizer = transforms.Compose([
                transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])
            ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629],
                                    [1 / 0.2672, 1 / 0.2564, 1 / 0.2629])
        ])

    elif dataset_name == 'tiny':
        normalizer = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                                 (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)),
        ])

    poison_transform = None
    trigger = None
    trigger_mask = None

    if poison_type in ['badnet', 'blend', 'lc', 'adaptive_blend', 'wanet']:

        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # trigger mask transform; remove `Normalize`!
        trigger_mask_transform_list = []
        for t in trigger_transform.transforms:
            if "Normalize" not in t.__class__.__name__:
                trigger_mask_transform_list.append(t)
        trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

        if trigger_name != 'none':
            trigger_path = os.path.join(attack_config.triggers_dir, trigger_name)
            # print('trigger : ', trigger_path)
            trigger = Image.open(trigger_path).convert("RGB")

            trigger_mask_path = os.path.join(attack_config.triggers_dir, 'mask_%s' % trigger_name)

            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = trigger_mask_transform(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                trigger_map = trigger_mask_transform(trigger)
                trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                                trigger_map[2] > 0).float()

            trigger = trigger_transform(trigger)
            # print('trigger_shape: ', trigger.shape)
            trigger_mask = trigger_mask

        if poison_type == 'badnet':
            from attack import badnet
            poison_transform = badnet.poison_transform(
                img_size=img_size, 
                trigger_mark=trigger,
                trigger_mask=trigger_mask, 
                target_label=target_label
            )

        elif poison_type == 'blend':
            from attack import blend
            poison_transform = blend.poison_transform(
                img_size=img_size, 
                trigger=trigger,
                target_label=target_label, 
                alpha=alpha
            )

        elif poison_type == 'lc':
            if args.dataset not in ['cifar10']:
                raise Exception("Dataset is not implemented")
            from attack import lc
            poison_transform = lc.poison_transform(
                img_size=img_size, 
                trigger_mark=trigger,
                trigger_mask=trigger_mask,
                target_label=target_label
            )

        elif poison_type == 'wanet':
            s = 0.5
            k = 4
            grid_rescale = 1
            path = os.path.join(get_poison_set_dir(args), 'identity_grid.pt')
            identity_grid = torch.load(path)
            path = os.path.join(get_poison_set_dir(args), 'noise_grid.pt')
            noise_grid = torch.load(path)

            from attack import wanet
            poison_transform = wanet.poison_transform(
                img_size=img_size, 
                denormalizer=denormalizer,
                identity_grid=identity_grid, 
                noise_grid=noise_grid, 
                s=s, 
                k=k,
                grid_rescale=grid_rescale, 
                normalizer=normalizer,
                target_label=target_label
            )

        elif poison_type == 'adaptive_blend':

            from attack import adaptive_blend
            poison_transform = adaptive_blend.poison_transform(
                img_size=img_size, 
                trigger=trigger,
                target_label=target_label,
                alpha=args.alpha if args.test_alpha is None else args.test_alpha
            )

        return poison_transform
    else:
        raise NotImplementedError('<Undefined> Poison_Type = %s' % poison_type)

def get_info_dataset(args, get_count=False):
    if args.dataset == 'cifar10':
        img_size = 32
        num_classes = 10
        num_train_imgs = 50000
        num_test_imgs = 10000
    elif args.dataset == 'gtsrb':
        img_size = 32
        num_classes = 43
        num_train_imgs = 39209
        num_test_imgs = 12630
    elif args.dataset == 'tiny':
        img_size = 64
        num_classes = 200
        num_train_imgs = 100000
        num_val_imgs = 10000
        num_test_imgs = 10000
    
    if get_count:
        if args.dataset != 'tiny':
            return num_train_imgs, num_test_imgs
        else:
            return num_train_imgs, num_val_imgs, num_test_imgs
    else:
        return img_size, num_classes


def get_clean_test_dir(args):
    if args.dataset != 'tiny':
        dir = os.path.join(common_config.clean_data_dir, args.dataset)
    else:
        dir = os.path.join(common_config.clean_data_dir, args.dataset, 'val')
    
    return dir

