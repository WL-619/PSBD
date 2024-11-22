import argparse
import os, sys
sys.path.append('../')
import logging
logger = logging.getLogger(__name__)
from config import attack_config, common_config, default_args
from utils import tools, supervisor
from sklearn import metrics
import torch
from torchvision import transforms


def add_arguments(parser):
    """
        Args:
            dataset (str): The dataset we used
            
            poison_type (str): Attack types

            poisoning_ratio (float): Proportion of backdoor data in the entire training dataset
            
            cover_rate (float): Ratio of data samples with backdoor trigger but without label modification to target class

            alpha (float): Blend rate for blend or adaptive_blend attacks

            test_alpha (float): Blend rate for blend or adaptive_blend attacks during test stage

            trigger (str): trigger of attacks

            no_aug (bool, default=False): Whether to use data augmentation. If True, data augmentation will be applied

            no_normalize (bool, default=False): Whether to use data normalization. If True, data normalization will be applied

            load_bench_data (bool, default=False): Whether to use data provided by https://github.com/SCLBD/BackdoorBench

            baseline (str): Selection of baseline: ss(spectral signature), strip, spectre, scan, scp    

            random_seed (int): Random seed for model training, rather than the seed used for generating the poisoned dataset

            result_path (str): Path to save result files

            checkpoint_save_path (str): Path to save checkpoints for loading checkpoints

            val_budget_rate (float, default=0.05): Proportion of clean extra validation data compared to the entire poisoned training set

            select_model (int, default=95): Model used for backdoor data detection

            load_checkpoint_path (str): Directly specify the path of checkpoints

    """

    parser.add_argument('-dataset', type=str, required=False,
                        default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'])
    parser.add_argument('-poison_type', type=str, required=False,
                        choices=default_args.parser_choices['poison_type'])
    parser.add_argument('-poisoning_ratio', type=float,  required=False,
                        choices=default_args.parser_choices['poisoning_ratio'],
                        default=default_args.parser_default['poisoning_ratio'])
    parser.add_argument('-cover_rate', type=float, required=False,
                        choices=default_args.parser_choices['cover_rate'],
                        default=default_args.parser_default['cover_rate'])
    parser.add_argument('-alpha', type=float, required=False,
                        default=default_args.parser_default['alpha'])
    parser.add_argument('-test_alpha', type=float, required=False, default=None)
    parser.add_argument('-trigger', type=str, required=False,default=None)
    parser.add_argument('-no_aug', default=False, action='store_true')
    parser.add_argument('-no_normalize', default=False, action='store_true')
    parser.add_argument('-load_bench_data', default=False, action='store_true')
    parser.add_argument('-baseline', type=str)
    parser.add_argument('-random_seed', type=int, required=False, default=default_args.random_seed)
    parser.add_argument('-result_path', type=str, default='./result')
    parser.add_argument('-checkpoint_save_path', type=str, required=False, default=default_args.checkpoint_save_path)
    parser.add_argument('-val_budget_rate', type=float, default=0.05)
    parser.add_argument('-select_model', type=int, default=95)
    parser.add_argument('-load_checkpoint_path', type=str, default=None)
    return parser

def prepare_dataset(args):
    if args.trigger is None and not args.load_bench_data:
        args.trigger = attack_config.trigger_default[args.dataset][args.poison_type]

    if args.load_bench_data:
        attack_config.record_poison_seed = False
    else:
        attack_config.record_poison_seed = True

    extra_val_path = os.path.join(common_config.extra_data_dir, args.dataset, 'extra_val_split', 'val_budget_rate_'+str(args.val_budget_rate), 'extra_val_data.pt')
    extra_val_set = torch.load(extra_val_path)

    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    poison_set_dir = supervisor.get_poison_set_dir(args)
    if not args.load_bench_data:
        clean_train_imgs_path = os.path.join(poison_set_dir, 'clean_imgs.pt')
        clean_train_labels_path = os.path.join(poison_set_dir, 'clean_labels.pt')
        backdoor_train_imgs_path = os.path.join(poison_set_dir, 'backdoor_imgs.pt')
        backdoor_train_labels_path = os.path.join(poison_set_dir, 'backdoor_labels.pt')
        clean_train_set = tools.IMG_Dataset(
            data_dir=clean_train_imgs_path,
            label_path=clean_train_labels_path,
            transforms=data_transform
        )
        
        backdoor_train_set = tools.IMG_Dataset(
            data_dir=backdoor_train_imgs_path,
            label_path=backdoor_train_labels_path,
            transforms=data_transform
        )
            
        if args.poison_type in ["wanet", "adaptive_blend"]:
            cover_train_imgs_path = os.path.join(poison_set_dir, 'cover_imgs.pt')
            cover_train_labels_path = os.path.join(poison_set_dir, 'cover_labels.pt')
            cover_train_set = tools.IMG_Dataset(
                data_dir=cover_train_imgs_path,
                label_path=cover_train_labels_path,
                transforms=data_transform
            )

    else:
        clean_train_set = torch.load(os.path.join(poison_set_dir, 'clean_train_data.pt'))
        backdoor_train_set = torch.load(os.path.join(poison_set_dir, 'backdoor_train_data.pt'))
        transform = data_transform
        transformed_clean_train_imgs = []
        transformed_clean_train_labels = []

        for idx in range(len(clean_train_set)):
            sample, target = clean_train_set[idx]
            transformed_sample = transform(sample)
            transformed_clean_train_imgs.append(transformed_sample)
            transformed_clean_train_labels.append(target)

        transformed_clean_train_imgs = torch.stack(transformed_clean_train_imgs, dim=0)
        transformed_clean_train_labels = torch.tensor(transformed_clean_train_labels)

        transformed_backdoor_train_imgs = []
        transformed_backdoor_train_labels = []
        transform = transforms.Compose([t for t in transform.transforms if not isinstance(t, transforms.ToTensor)])

        for idx in range(len(backdoor_train_set)):
            sample, target = backdoor_train_set[idx]
            transformed_sample = transform(sample)
            transformed_backdoor_train_imgs.append(transformed_sample)
            transformed_backdoor_train_labels.append(attack_config.target_label[args.dataset])
        
        transformed_backdoor_train_imgs = torch.stack(transformed_backdoor_train_imgs, dim=0)
        transformed_backdoor_train_labels = torch.tensor(transformed_backdoor_train_labels)

        clean_train_set = torch.utils.data.TensorDataset(transformed_clean_train_imgs, transformed_clean_train_labels)
        backdoor_train_set = torch.utils.data.TensorDataset(transformed_backdoor_train_imgs, transformed_backdoor_train_labels)

    if args.poison_type not in ["wanet", "adaptive_blend"]:
        poisoned_train_set = torch.utils.data.ConcatDataset([clean_train_set, backdoor_train_set])
    else:
        poisoned_train_set = torch.utils.data.ConcatDataset([clean_train_set, cover_train_set, backdoor_train_set])

    # we do not consider cover data as backdoor data because cover data cannot trigger the backdoor attack
    # we treat backdoor data as positive samples
    backdoor_indicator = torch.zeros(len(poisoned_train_set))
    backdoor_indicator[-len(backdoor_train_set):] = 1

    return poisoned_train_set, extra_val_set, backdoor_indicator

def get_checkpoint_path(args):
    if args.load_checkpoint_path is None:
        if args.load_bench_data:
            checkpoint_path = os.path.join(args.checkpoint_save_path, "data_from_bench")
        else:
            checkpoint_path = args.checkpoint_save_path
            
        checkpoint_path = os.path.join(checkpoint_path, arch.__name__, args.dataset)

        if args.no_aug:
            prefix = 'no_aug_'
        else:
            prefix = 'has_aug_'

        if args.no_normalize:
            prefix = prefix + 'no_norm'
        else:
            prefix = prefix + 'has_norm'

        checkpoint_path = os.path.join(checkpoint_path, prefix)

        checkpoint_path = os.path.join(checkpoint_path, str(args.poisoning_ratio), 'random_seed_'+str(args.random_seed), args.poison_type)

        if args.poison_type in ['blend', 'adaptive_blend']:
            checkpoint_path =  checkpoint_path + '_alpha_' + str(args.alpha)

        checkpoint_path = os.path.join(checkpoint_path, 'target_label_'+str(attack_config.target_label[args.dataset])+'_seed_'+str(attack_config.poison_seed))
    else:
        checkpoint_path = args.load_checkpoint_path
    return checkpoint_path


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    tools.setup_seed(args.random_seed)

    img_size, num_classes = supervisor.get_info_dataset(args)
    arch = common_config.arch[args.dataset]

    ckpt_path = get_checkpoint_path(args)
    logger.info(f"Loading selected model: {ckpt_path+'/epoch_'+str(args.select_model)+'.pt'}")
    ckpt = torch.load(ckpt_path+'/epoch_'+str(args.select_model)+'.pt')
    selcected_model = arch(num_classes=num_classes)
    selcected_model.load_state_dict(ckpt)
    selcected_model = selcected_model.cuda()
    selcected_model.eval()

    poisoned_train_set, extra_val_set, backdoor_indicator = prepare_dataset(args)

    from detection import scan, strip, spectral_signature, spectre_python, scp, cdl

    if args.baseline == 'scan':
        suspicious_indices = scan.cleanser(poisoned_train_set, selcected_model, num_classes, extra_val_set)
    elif args.baseline == 'strip':
        suspicious_indices = strip.cleanser(poisoned_train_set, selcected_model, args, extra_val_set)
    elif args.baseline == 'spectre':
        suspicious_indices = spectre_python.cleanser(poisoned_train_set, selcected_model, num_classes, args, extra_val_set)
    elif args.baseline == 'ss':
        suspicious_indices = spectral_signature.cleanser(poisoned_train_set, selcected_model, num_classes, args)
    elif args.baseline == 'scp':
        suspicious_indices = scp.cleanser(poisoned_train_set, selcected_model, args, extra_val_set)
    elif args.baseline == 'cdl':
        suspicious_indices = cdl.cleanser(poisoned_train_set, selcected_model, args, extra_val_set)
    else:
        raise NotImplementedError('Baseline is not implemented')

    # we treat backdoor data as positive samples.
    if args.baseline != 'cdl':
        backdoor_prediction = torch.zeros(len(poisoned_train_set))
        backdoor_prediction[suspicious_indices] = 1
    else:
        backdoor_prediction = suspicious_indices

    tn, fp, fn, tp = metrics.confusion_matrix(backdoor_indicator, backdoor_prediction).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    logger.info("\n\nPerformance:")
    logger.info("TPR: {:.3f}".format(tpr))
    logger.info("FPR: {:.3f}".format(fpr))