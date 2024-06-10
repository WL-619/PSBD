import argparse
import os, sys
sys.path.append('../')
from tqdm import tqdm
from config import attack_config, common_config, default_args
from utils import tools, supervisor
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import datetime
import json
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn import metrics
from models import resnet_drop

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

            log (bool, default=False): Whether to save log file

            baseline (str): Selection of baseline

            random_seed (int): Random seed for model training and dropout, rather than the seed used for generating the poisoned dataset

            result_path (str): Path to save result files

            checkpoint_save_path (str): Path to save checkpoints for loading checkpoints

            val_budget_rate (float, default=0.05): Proportion of clean extra validation data compared to the entire poisoned training set

            select_model (int, default=95): Model used for backdoor data detection

            load_checkpoint_path (str): Directly specify the path of checkpoints

            fig_path (str): Path to save figs

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
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-random_seed', type=int, required=False, default=default_args.random_seed)
    parser.add_argument('-checkpoint_save_path', type=str, required=False, default=default_args.checkpoint_save_path)
    parser.add_argument('-val_budget_rate', type=float, default=0.05)
    parser.add_argument('-select_model', type=int, default=95)
    parser.add_argument('-load_checkpoint_path', type=str, default=None)
    parser.add_argument('-fig_path', type=str, default=default_args.fig_save_path)
    return parser

def get_dataloader(args):
    if args.trigger is None and not args.load_bench_data:
        args.trigger = attack_config.trigger_default[args.dataset][args.poison_type]

    if args.load_bench_data:
        attack_config.record_poison_seed = False
    else:
        attack_config.record_poison_seed = True

    if args.dataset == 'tiny':
        kwargs = {'num_workers': 32, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    
    training_setting = common_config.training_setting
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    # clean extra validation data
    extra_val_path = os.path.join(common_config.extra_data_dir, args.dataset, 'extra_val_split', 'val_budget_rate_'+str(args.val_budget_rate), 'extra_val_data.pt')
    extra_val_set = torch.load(extra_val_path)
    logger.info("Get extra val loader...")
    extra_val_set_loader = torch.utils.data.DataLoader(
            extra_val_set,
            batch_size=training_setting['batch_size'],
            shuffle=False,
            drop_last=False,
            worker_init_fn=tools.worker_init,
            **kwargs
    )

    poison_set_dir = supervisor.get_poison_set_dir(args)
    if not args.load_bench_data:
        clean_train_imgs_path = os.path.join(poison_set_dir, 'clean_imgs.pt')
        clean_train_labels_path = os.path.join(poison_set_dir, 'clean_labels.pt')
        backdoor_train_imgs_path = os.path.join(poison_set_dir, 'backdoor_imgs.pt')
        backdoor_train_labels_path = os.path.join(poison_set_dir, 'backdoor_labels.pt')
        
        # clean training data
        clean_train_set = tools.IMG_Dataset(
            data_dir=clean_train_imgs_path,
            label_path=clean_train_labels_path,
            transforms=data_transform
        )
        
        # backdoor training data
        backdoor_train_set = tools.IMG_Dataset(
            data_dir=backdoor_train_imgs_path,
            label_path=backdoor_train_labels_path,
            transforms=data_transform
        )
            
        if args.poison_type in ["wanet", "adaptive_blend"]:
            cover_train_imgs_path = os.path.join(poison_set_dir, 'cover_imgs.pt')
            cover_train_labels_path = os.path.join(poison_set_dir, 'cover_labels.pt')

            # cover training data
            cover_train_set = tools.IMG_Dataset(
                data_dir=cover_train_imgs_path,
                label_path=cover_train_labels_path,
                transforms=data_transform
            )
    else:
        # if using bench data, manual transformation is required
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

    logger.info("Get poisoned set loader...")
    poisoned_train_set_loader = torch.utils.data.DataLoader(
            poisoned_train_set,
            batch_size=training_setting['batch_size'], 
            shuffle=False, 
            worker_init_fn=tools.worker_init, 
            **kwargs
        )
    logger.info("Get clean loader...")
    clean_loader = torch.utils.data.DataLoader(
            clean_train_set,
            batch_size=training_setting['batch_size'],
            shuffle=False,
            drop_last=False,
    )
    logger.info("Get backdoor loader...")
    backdoor_loader = torch.utils.data.DataLoader(
            backdoor_train_set,
            batch_size=training_setting['batch_size'],
            shuffle=False,
            drop_last=False,
    )

    # we do not consider cover data as backdoor data because cover data cannot trigger the backdoor attack
    # we treat backdoor data as positive samples
    backdoor_indicator = torch.zeros(len(poisoned_train_set))
    backdoor_indicator[-len(backdoor_train_set):] = 1

    return poisoned_train_set_loader, clean_loader, backdoor_loader, extra_val_set_loader, backdoor_indicator
    
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

def compute_psu(args, model, dataloader, mode, forward_passes=3, drop_p=0.8, return_shift=False):
    with torch.no_grad():
        tools.setup_seed(args.random_seed)
        with tqdm(total=len(dataloader)) as t:
            t.set_description(desc=f'Compute ' + mode + ' Uncertainty')
            psu = []
            ps = []
            
            for idx, (inputs, target) in enumerate(dataloader):
                inputs = inputs.cuda()
                target = target.cuda()

                # compute prediction confidences before turning on dropout
                model.eval()
                primitive_prob = F.softmax(model(inputs), dim=1)
                primitive_label = torch.argmax(primitive_prob, dim=1)                
                psu_per_batch = []

                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                        m.p = drop_p

                # compute prediction confidences after turning on dropout
                drop_prob = []
                for _ in range(forward_passes):
                    outputs = model(inputs)
                    outputs_idx = torch.argmax(F.softmax(outputs, dim=1), dim=1)

                    ps.append(outputs_idx[outputs_idx!=primitive_label])
                    drop_prob.append(F.softmax(outputs, dim=1))
                    
                drop_prob = torch.stack(drop_prob)
                mean_prob = torch.mean(drop_prob, dim=0)

                diff = primitive_prob - mean_prob

                # psu is the difference that only corresponds to the primitive label
                psu_per_batch.append(diff.gather(1, primitive_label.view(-1,1)).squeeze(1))
                psu_per_batch = torch.stack(psu_per_batch)
                psu_per_batch = psu_per_batch.squeeze(0)
                
                psu.append(psu_per_batch)
                t.update(1)
                         
            psu = torch.cat(psu)
        
        ps = torch.cat(ps)

        # proportion of times ps occurs out of all inference occurrences
        shift_ratio = len(ps)/(len(dataloader.dataset)*forward_passes)
        logger.info(f"\nshift ratio = {shift_ratio*100}%\n")

    if return_shift:
        return ps, shift_ratio
    else:
        return psu

def get_psu(model, clean_loader, bd_loader, val_loader, forward_passes=3, drop_p=0.8):    
    # compute psu of clean training data, backdoor training data, and clean extra validation data
    clean_psu = torch.empty(0).cuda()
    backdoor_psu = torch.empty(0).cuda()
    val_psu = torch.empty(0).cuda()
    
    model = model.cuda()

    c_psu = compute_psu(args, model, clean_loader, 'clean', forward_passes, drop_p)
    b_psu = compute_psu(args, model, bd_loader, 'backdoor', forward_passes, drop_p)
    v_psu = compute_psu(args, model, val_loader, 'val', forward_passes, drop_p)
    
    if val_psu.numel() == 0:
        clean_psu = c_psu.unsqueeze(1)
        backdoor_psu = b_psu.unsqueeze(1)
        val_psu = v_psu.unsqueeze(1)

    else:
        clean_psu = torch.cat((clean_psu, c_psu.unsqueeze(1)), dim=1)
        backdoor_psu = torch.cat((backdoor_psu, b_psu.unsqueeze(1)), dim=1)
        val_psu = torch.cat((val_psu, v_psu.unsqueeze(1)), dim=1)
            
    return clean_psu, backdoor_psu, val_psu

def select_dropout_rate(args, selcected_model, poisoned_train_set_loader, clean_loader, backdoor_loader, extra_val_set_loader, draw_shift_ratio=False):
    # dropout rate chosen for psbd
    p_index = [i / 10.0 for i in range(1, 10)]
    total_ratio = []
    clean_ratio = []
    bd_ratio = []
    val_ratio = []

    for p in tqdm(p_index):
        logger.info(f"Compute psu at drop {p}...")
        _, t = compute_psu(args, selcected_model, poisoned_train_set_loader, 'total train', drop_p=p, return_shift=True)
        _, c = compute_psu(args, selcected_model, clean_loader, 'clean train', drop_p=p, return_shift=True)
        _, b = compute_psu(args, selcected_model, backdoor_loader, 'backdoor', drop_p=p, return_shift=True)
        _, v = compute_psu(args, selcected_model, extra_val_set_loader, 'val', drop_p=p, return_shift=True)
        total_ratio.append(t)
        clean_ratio.append(c)
        bd_ratio.append(b)
        val_ratio.append(v)

    diff = []
    start = 0
    for i in range(len(p_index)):
        if(val_ratio[i] >= 0.8):
            if start == 0:
                start = p_index[i]
            diff.append(val_ratio[i] - total_ratio[i])

    if len(diff) == 0:
        logger.info('None val_ratio >= 0.8, use max(val_ratio - total_ratio)!')
        selected_p = np.argmax(np.array(val_ratio)-np.array(total_ratio))
        selected_p = (selected_p + 1) * 0.1
    else:
        selected_p = np.argmax(diff)
        selected_p = start + selected_p * 0.1
    
    if draw_shift_ratio:
        shift_ratio_fig(p_index, total_ratio, clean_ratio, bd_ratio, val_ratio, selected_p)

    logger.info(f"Selected p is {selected_p}")
    return selected_p

def box_fig(clean_psu, backdoor_psu, val_psu):
    plt.boxplot([clean_psu.numpy(), backdoor_psu.numpy(), 
                val_psu.numpy()], tick_labels=['clean', 'backdoor', 'val'],
               boxprops=dict(linewidth=2.0),
               whiskerprops=dict(linewidth=2.0))
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    save_path = os.path.join(args.fig_path, 'box_fig')
    if not os.path.exists(save_path): tools.create_missing_folders(save_path)
    plt.savefig(save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf')
    logger.info(f"Box fig of all data types saved at {save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf'}")
    plt.clf()
    return

def ps_intensity_fig(num_classes, clean_train_ps, bd_train_ps, extra_clean_ps):
    width = 0.3
    plt.figure(figsize=(13, 7))
    if num_classes <= 10:
        x_tick = [i for i in range(0, num_classes)]
    else:
        x_tick = []

    # display only the top k results with the highest ps intensity on the tiny imageNet dataset
    k = 3

    # ps intensity of clean training data
    pre, counts = torch.unique(clean_train_ps, return_counts=True)
    if num_classes > 10:
        if len(counts) < k:
            logger.info('top PS intensity is less than k! set the k = len(counts)')
            k = len(counts)
        counts, top_indices = torch.topk(counts, k)
        pre = pre[top_indices]
        intensity = counts / len(clean_train_ps)
        
        plt.bar(pre - width, intensity, width=width, label='clean')
        for i in pre.numpy():
            x_tick.append(i)
    else:
        intensity = counts / len(clean_train_ps)
        max_index = np.argmax(intensity)
        max_intensity = intensity[max_index]
        plt.bar(pre[max_index] - width, max_intensity, width=width, label='clean')
        x_tick.append(pre[max_index])
    

    # ps intensity of backdoor training data
    pre, counts = torch.unique(bd_train_ps, return_counts=True)

    if num_classes > 10:
        if len(counts) < k:
            logger.info('top PS intensity is less than k! set the k = len(counts)')
            k = len(counts)
        counts, top_indices = torch.topk(counts, k)
        pre = pre[top_indices]
        intensity = counts / len(bd_train_ps)

        plt.bar(pre, intensity, width=width, label='backdoor')
        for i in pre.numpy():
            x_tick.append(i)

    else:
        intensity = counts / len(bd_train_ps)
        max_index = np.argmax(intensity)
        max_intensity = intensity[max_index]
        plt.bar(pre[max_index], max_intensity, width=width, label='backdoor')
        x_tick.append(pre[max_index])


    # ps intensity of clean extra validation data
    pre, counts = torch.unique(extra_clean_ps, return_counts=True)

    if num_classes > 10:
        if len(counts) < k:
            logger.info('top PS intensity is less than k! set the k = len(counts)')
            k = len(counts)
        counts, top_indices = torch.topk(counts, k)
        pre = pre[top_indices]
        intensity = counts / len(extra_clean_ps)
        
        plt.bar(pre + width, intensity, width=width, label='validation')
        for i in pre.numpy():
            x_tick.append(i)

    else:
        intensity = counts / len(extra_clean_ps)
        max_index = np.argmax(intensity)
        max_intensity = intensity[max_index]
        plt.bar(pre[max_index] + width, max_intensity, width=width, label='validation')
        x_tick.append(pre[max_index])

    plt.xlabel('Prediction Shift', fontsize=50, fontweight='bold')
    plt.xticks(x_tick, size=50)

    plt.ylabel('Shift Intensity', fontsize=50, fontweight='bold', loc='top')
    plt.ylim(0, 1+0.09)
    plt.yticks(size=50)

    plt.legend(loc='upper right', frameon=False, fontsize=35)
    plt.tight_layout()

    save_path = os.path.join(args.fig_path, 'prediction_shift_intensity')
    if not os.path.exists(save_path): tools.create_missing_folders(save_path)
    plt.savefig(save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf')
    logger.info(f"Prediction shift intensity saved at {save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf'}")
    plt.clf()
    return

def smooth(x, y):
    # function to generate interpolation for smooth curves
    x = np.array(x)
    y = np.array(y)
    t = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y)
    smooth_y = spl(t)
    return t, smooth_y

def shift_ratio_fig(drop_p, total_ratio, clean_ratio, bd_ratio, val_ratio, selected_p):
    # shift ratio curve
    figure, ax = plt.subplots(1, 1, figsize=(13, 7))
    patch = ax.patch
    patch.set_color("#EBEBEB")
    t, smooth_y = smooth(drop_p, total_ratio)
    ax.plot(t, smooth_y, label='clean+backdoor', alpha=0.8, linestyle='--', color='black', linewidth=6)

    t, smooth_y = smooth(drop_p, clean_ratio)
    ax.plot(t, smooth_y, label='clean', alpha=0.8, color='#1f77b4', linewidth=6)

    t, smooth_y = smooth(drop_p, bd_ratio)
    ax.plot(t, smooth_y, label='backdoor', alpha=0.8, color='#ff7f0e', linewidth=6)

    t, smooth_y = smooth(drop_p, val_ratio)
    ax.plot(t, smooth_y, label='validation', alpha=0.8, color='#2ca02c', linewidth=6)

    ax.axvline(x=selected_p, color='#800080', linestyle='-.', linewidth=6)

    ax.set_xlabel('Dropout Rate', fontsize=45, fontweight='bold')
    ax.set_xticks(drop_p)

    ax.set_ylabel('Shift Ratio', fontsize=45, fontweight='bold')
    ax.tick_params(axis='x', labelsize=45)
    ax.tick_params(axis='y', labelsize=45)

    ax.set_ylim(0, 1+0.09)

    ax.legend(loc='lower right', frameon=False, fontsize=35)

    figure.tight_layout()
    save_path = os.path.join(args.fig_path, 'shift_ratio')
    if not os.path.exists(save_path): tools.create_missing_folders(save_path)
    figure.savefig(save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf')
    plt.clf()
    logger.info(f"Shift ratio curve saved at {save_path + '/' + args.dataset + '_' + args.poison_type + '_pr_' + str(args.poisoning_ratio) + '.pdf'}")
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

    exe_start_time = datetime.datetime.now()
    
    poisoned_train_set_loader, clean_loader, backdoor_loader, \
    extra_val_set_loader, backdoor_indicator = get_dataloader(args)

    arch = resnet_drop.ResNet18
    ckpt_path = get_checkpoint_path(args)
    ckpt = torch.load(ckpt_path+'/epoch_'+str(args.select_model)+'.pt')

    img_size, num_classes = supervisor.get_info_dataset(args)

    logger.info("Loading selected model...")
    selcected_model = arch(num_classes=num_classes)
    selcected_model.load_state_dict(ckpt)
    selcected_model = selcected_model.cuda()

    logger.info("Get p...")
    selected_p = select_dropout_rate(
            args, 
            selcected_model, 
            poisoned_train_set_loader,
            clean_loader, 
            backdoor_loader, 
            extra_val_set_loader, 
            draw_shift_ratio=True
        )
    
    extra_val_psu = compute_psu(args, selcected_model, extra_val_set_loader, 'extra_val', drop_p=selected_p)
    train_set_psu = compute_psu(args, selcected_model, poisoned_train_set_loader, 'train', drop_p=selected_p)
    
    threshold = extra_val_psu.quantile(0.25)
    backdoor_prediction = train_set_psu < threshold
    backdoor_prediction = backdoor_prediction.to(torch.int).cpu()

    tn, fp, fn, tp = metrics.confusion_matrix(backdoor_indicator, backdoor_prediction).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    print("\n\nPerformance:")
    print("TPR: {:.3f}".format(tpr))
    print("FPR: {:.3f}".format(fpr))

    # clean_psu, backdoor_psu, val_psu = get_psu(
    #         selcected_model, 
    #         clean_loader,
    #         backdoor_loader,
    #         extra_val_set_loader,
    #         forward_passes=3, 
    #         drop_p=selected_p
    #     )
    # clean_psu = clean_psu.squeeze(1)
    # backdoor_psu = backdoor_psu.squeeze(1)
    # val_psu = val_psu.squeeze(1)
    # box_fig(clean_psu.cpu(), backdoor_psu.cpu(), val_psu.cpu())
    
    # c_ps, _ = compute_psu(args, selcected_model, clean_loader, 'clean', forward_passes=3, drop_p=selected_p, return_shift=True)
    # b_ps, _ = compute_psu(args, selcected_model, backdoor_loader, 'backdoor', forward_passes=3, drop_p=selected_p, return_shift=True)
    # v_ps, _ = compute_psu(args, selcected_model, extra_val_set_loader, 'val', forward_passes=3, drop_p=selected_p, return_shift=True)
    # ps_intensity_fig(num_classes, c_ps.cpu(), b_ps.cpu(), v_ps.cpu())
    
    # log information
    if args.log:
        exe_end_time = datetime.datetime.now()

        log_path = os.path.join('./results', args.dataset, 'pr_'+str(args.poisoning_ratio), args.poison_type, 'psbd')
        if not os.path.exists(log_path): tools.create_missing_folders(log_path)
        log_path = os.path.join(log_path, '%s' % (exe_start_time.strftime('%Y-%m-%d-%H:%M:%S_')+str(args.random_seed)))

        with open(log_path+".txt", "a") as file:
            file.write("\n\nExecution Start Time:\n\t"+exe_start_time.strftime('%Y-%m-%d %H:%M:%S'))

            file.write("\n\nArgs:")
            file.write("\n\t\t")
            file.write(json.dumps(args.__dict__))

            file.write("\n\nPoisoned Dataset:")
            file.write("\n\tdata path:\t" + supervisor.get_poison_set_dir(args))
            file.write("\n\tload_bench_data:\t" + str(args.load_bench_data))
            file.write("\n\tcheckpoint_path:\t" + ckpt_path)
            
            
            file.write("\n\nDetection Metric:")
            file.write("\n\tTPR:\t" + str(tpr))
            file.write("\n\tFPR:\t" + str(fpr))

            file.write("\n\nAttack Info:")
            file.write("\n\tpoisoning_type:\t" + str(args.poison_type))
            file.write("\n\tpoisoning_ratio:\t" + str(args.poisoning_ratio))
            file.write("\n\ttrigger:\t" + str(args.trigger))
            file.write("\n\nExecution End Time:\n\t"+ str(exe_end_time))
            file.write("\n\nExecution Total Time:\n\t"+ str(exe_end_time-exe_start_time))