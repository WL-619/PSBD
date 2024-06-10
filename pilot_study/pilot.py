import argparse
import os, sys
sys.path.append('../')
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from config import attack_config, common_config, default_args
from utils import tools, supervisor
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

            scale (float): amplification factor for input pixel values

            random_seed (int, default=0): Random seed for dropout

            result_path (str): Path to save result files

            checkpoint_save_path (str): Path to save checkpoints for loading checkpoints

            val_budget_rate (float, default=0.05): Proportion of clean extra validation data compared to the entire poisoned training set

            load_checkpoint_path (str): Directly specify the path of checkpoints

            fig_path (str): Path to save figs

            result_save_path (str): Path to save results

            num_model (int, default=100): the number of training epochs to observe

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
    parser.add_argument('-scale', type=float, default=1.)
    parser.add_argument('-random_seed', type=int, required=False, default=default_args.random_seed)
    parser.add_argument('-checkpoint_save_path', type=str, required=False, default=default_args.checkpoint_save_path)
    parser.add_argument('-val_budget_rate', type=float, default=0.05)
    parser.add_argument('-load_checkpoint_path', type=str, default=None)
    parser.add_argument('-fig_path', type=str, required=False, default=default_args.fig_save_path)
    parser.add_argument('-result_save_path', type=str, required=False, default='./results')
    parser.add_argument('-num_model', type=int, default=100)
    return parser

def prepare_dataset(args):
    if args.trigger is None and not args.load_bench_data:
        args.trigger = attack_config.trigger_default[args.dataset][args.poison_type]
    
    if args.load_bench_data:
        attack_config.record_poison_seed = False
    else:
        attack_config.record_poison_seed = True

    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    extra_val_path = os.path.join(common_config.extra_data_dir, args.dataset, 'extra_val_split', 'val_budget_rate_'+str(args.val_budget_rate), 'extra_val_data.pt')
    extra_val_set = torch.load(extra_val_path)
    
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
            # we do not consider cover data as backdoor data because cover data cannot trigger the backdoor attack
            clean_train_set = torch.utils.data.ConcatDataset([clean_train_set, cover_train_set])

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
            transformed_backdoor_train_labels.append(int(poison_set_dir.split('/')[-1].split('=')[-1]))
        
        transformed_backdoor_train_imgs = torch.stack(transformed_backdoor_train_imgs, dim=0)
        transformed_backdoor_train_labels = torch.tensor(transformed_backdoor_train_labels)

        clean_train_set = torch.utils.data.TensorDataset(transformed_clean_train_imgs, transformed_clean_train_labels)
        backdoor_train_set = torch.utils.data.TensorDataset(transformed_backdoor_train_imgs, transformed_backdoor_train_labels)

    return clean_train_set, backdoor_train_set, extra_val_set

def get_checkpoint_path(args):
    if args.load_checkpoint_path is None:
        if args.load_bench_data:
            checkpoint_path = os.path.join(args.checkpoint_save_path, "data_from_bench")
        else:
            checkpoint_path = default_args.checkpoint_save_path
            
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

def mc_drop_uncertainty(args, model, dataloader, mode, forward_passes=3, p=0.2):
    # based on https://github.com/nayeemrizve/ups/blob/3ba73f3/utils/pseudo_labeling_util.py#L37
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)
    model.eval()
    tools.setup_seed(args.random_seed)
    tools.enable_dropout(model, p)
    with tqdm(total=len(dataloader)) as t:
        t.set_description(desc=f'Compute ' + mode + ' MC-Dropout Uncertainty')
        total_uncertainty = []
        with torch.no_grad():
            for _, (inputs, target) in enumerate(dataloader):
                inputs = inputs.cuda()
                target = target.cuda()
                out_prob = []
                
                if args.scale > 1:
                    inputs = normalizer(torch.clip(denormalizer(inputs) * args.scale, 0.0, 1.0))
                
                for _ in range(forward_passes):
                    outputs = model(inputs)
                    out_prob.append(F.softmax(outputs, dim=1))

                out_prob = torch.stack(out_prob)
                out_std = torch.std(out_prob, dim=0)

                out_prob = torch.mean(out_prob, dim=0)
                _, max_idx = torch.max(out_prob, dim=1)
                max_std = out_std.gather(1, max_idx.view(-1,1))
                total_uncertainty.append(max_std.squeeze(1))
                t.update(1)
        uncertainty = torch.cat(total_uncertainty)
        
        t.set_postfix(uncertainty=f"{torch.mean(uncertainty):.3e}")
    return uncertainty

def uncertainty_fig(args, clean_un, backdoor_un, val_un, epochs):
    epoch = range(1, epochs+1, 1)

    plt.figure(figsize=(9, 7))
    plt.plot(epoch, clean_un, label='clean', marker='^')
    plt.plot(epoch, backdoor_un, label='backdoor', marker='*', linestyle='dashed')
    plt.plot(epoch, val_un, label='validation', marker='o')

    x_index = [1, 20, 40, 60, 80, 100]
    plt.xticks(ticks=x_index, labels=x_index, size=30)
    plt.yticks(size=30)
    plt.ylim(bottom=min(min(clean_un), min(backdoor_un), min(val_un))-0.05)
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Epoch', fontsize=32, fontweight='bold')
    plt.ylabel('MC-Dropout Uncertainty', fontsize=32, fontweight='bold', loc='top')

    plt.legend(fontsize=23, loc='upper right')

    plt.tight_layout()
    if args.scale > 1:
        figs_save_path = os.path.join(args.fig_path, 'pilot_2')
    else:
        figs_save_path = os.path.join(args.fig_path, 'pilot_1')
    
    figs_save_path = os.path.join(figs_save_path, args.dataset, args.poison_type, str(args.poisoning_ratio), 'scale_' + str(args.scale))
    if not os.path.exists(figs_save_path): tools.create_missing_folders(figs_save_path)
    figs_save_path = os.path.join(figs_save_path, 'random_seed_'+ str(args.random_seed) + '.pdf')

    plt.savefig(figs_save_path)
    logger.info(f"Fig saved at {figs_save_path}")
    plt.clf()
    return


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
    if args.dataset == 'tiny':
        kwargs = {'num_workers': 32, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4, 'pin_memory': True}

    from models import resnet_drop_for_pilot
    arch = resnet_drop_for_pilot.ResNet18


    training_setting = common_config.training_setting

    clean_train_set, backdoor_train_set, extra_val_set = prepare_dataset(args)

    clean_train_set_loader = torch.utils.data.DataLoader(
        clean_train_set,
        batch_size=training_setting['batch_size'], 
        shuffle=False,
        drop_last=False, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )

    backdoor_train_set_loader = torch.utils.data.DataLoader(
        backdoor_train_set,
        batch_size=training_setting['batch_size'], 
        shuffle=False,
        drop_last=False, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )

    extra_val_set_loader = torch.utils.data.DataLoader(
            extra_val_set,
            batch_size=training_setting['batch_size'],
            shuffle=False,
            drop_last=False,
            worker_init_fn=tools.worker_init,
            **kwargs
    )

    ckpt_path = get_checkpoint_path(args)

    clean_un = torch.empty(0).cuda()
    backdoor_un = torch.empty(0).cuda()
    val_un = torch.empty(0).cuda()

    for i in range(1, args.num_model+1):
        ckpt = torch.load(ckpt_path+'/epoch_'+str(i)+'.pt')
        model = arch(num_classes=num_classes)
        model.load_state_dict(ckpt)
        model = model.cuda()
        
        logger.info("\nloading model: " + ckpt_path + '/epoch_'+str(i)+'.pt' + "\n")

        c_un = mc_drop_uncertainty(args, model, clean_train_set_loader, 'clean train data')
        b_un = mc_drop_uncertainty(args, model, backdoor_train_set_loader, 'backdoor train data')
        v_un = mc_drop_uncertainty(args, model, extra_val_set_loader, 'clean extra_val data')

        if val_un.numel() == 0:
                clean_un = c_un.unsqueeze(1)
                backdoor_un = b_un.unsqueeze(1)
                val_un = v_un.unsqueeze(1)
        else:
            clean_un = torch.cat((clean_un, c_un.unsqueeze(1)), dim=1)
            backdoor_un = torch.cat((backdoor_un, b_un.unsqueeze(1)), dim=1)
            val_un = torch.cat((val_un, v_un.unsqueeze(1)), dim=1)

    uncertainty_fig(
        args, 
        clean_un.mean(dim=0).cpu().numpy(), 
        backdoor_un.mean(dim=0).cpu().numpy(), 
        val_un.mean(dim=0).cpu().numpy(), 
        args.num_model
    )
