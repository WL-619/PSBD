import argparse
import os, sys
sys.path.append('../')
import time
from tqdm import tqdm
from config import attack_config, common_config, default_args
from utils import tools, supervisor
import json
from torchvision import datasets, transforms
from torch import nn
import torch
import datetime


def add_arguments(parser):
    """
        Args:
            dataset (str): The dataset we used
            
            poison_type (str): Attack types

            poisoning_ratio (float): Proportion of backdoor data in the entire training dataset

            cover_rate (float): Ratio of data samples with backdoor trigger but without label modification to target class

            alpha (float): Blend rate for blend or adaptive_blend attacks

            test_alpha (float): Blend rate for blend or adaptive_blend attacks during test stage

            resume (int):

            resume_from_meta_info (bool):

            trigger (str): trigger of attacks

            no_aug (bool, default=False): Whether to use data augmentation. If True, data augmentation will be applied

            no_normalize (bool, default=False): Whether to use data normalization. If True, data normalization will be applied

            load_bench_data (bool, default=False): Whether to use data provided by https://github.com/SCLBD/BackdoorBench

            log (bool, default=False): Whether to save log file

            random_seed (int): Random seed for model training and dropout, rather than the seed used for generating the poisoned dataset

            checkpoint_save_path (str): Path to save checkpoints for loading checkpoints

            checkpoint_save_epoch (int, default=1): Save model if current epoch >= checkpoint_save_epoch

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
    parser.add_argument('-resume', type=int, required=False, default=0)
    parser.add_argument('-resume_from_meta_info', default=False, action='store_true')
    parser.add_argument('-trigger', type=str, required=False,default=None)
    parser.add_argument('-no_aug', default=False, action='store_true')
    parser.add_argument('-no_normalize', default=False, action='store_true')
    parser.add_argument('-load_bench_data', default=False, action='store_true')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-random_seed', type=int, required=False, default=default_args.random_seed)
    parser.add_argument('-checkpoint_save_path', type=str, required=False, default=default_args.checkpoint_save_path)
    parser.add_argument('-checkpoint_save_epoch', type=int, required=False, default=default_args.checkpoint_save_epoch)
    return parser

def prepare_dataset(args):
    if args.trigger is None and not args.load_bench_data:
        args.trigger = attack_config.trigger_default[args.dataset][args.poison_type]

    if args.load_bench_data:
        attack_config.record_poison_seed = False
    else:
        attack_config.record_poison_seed = True

    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    clean_test_dir = supervisor.get_clean_test_dir(args)

    if args.dataset == 'cifar10':
        clean_test_set = datasets.CIFAR10(
            root=clean_test_dir, 
            transform=data_transform, 
            train=False
        )
    elif args.dataset == 'tiny':
        clean_test_set = datasets.ImageFolder(
            root=clean_test_dir, 
            transform=data_transform
        )
    else:
        raise NotImplementedError('Dataset is not implemented')

    poison_set_dir = supervisor.get_poison_set_dir(args)
    backdoor_test_set = None
    if not args.load_bench_data:
        clean_train_imgs_path = os.path.join(poison_set_dir, 'clean_imgs.pt')
        clean_train_labels_path = os.path.join(poison_set_dir, 'clean_labels.pt')
        backdoor_train_imgs_path = os.path.join(poison_set_dir, 'backdoor_imgs.pt')
        backdoor_train_labels_path = os.path.join(poison_set_dir, 'backdoor_labels.pt')
        clean_train_set = tools.IMG_Dataset(
            data_dir=clean_train_imgs_path,
            label_path=clean_train_labels_path,
            transforms=data_transform if args.no_aug else data_transform_aug
        )
        
        backdoor_train_set = tools.IMG_Dataset(
            data_dir=backdoor_train_imgs_path,
            label_path=backdoor_train_labels_path,
            transforms=data_transform if args.no_aug else data_transform_aug
        )

        if args.poison_type in ['lc'] and args.dataset == 'tiny':
            poison_test_dir = os.path.join(os.path.dirname(poison_set_dir), 'lc_backdoor_test')
            backdoor_test_imgs_path = os.path.join(poison_test_dir, 'backdoor_test_imgs.pt')
            backdoor_test_labels_path = os.path.join(poison_test_dir, 'backdoor_test_labels.pt')
            backdoor_test_set = tools.IMG_Dataset(
                data_dir=backdoor_test_imgs_path,
                label_path=backdoor_test_labels_path,
                transforms=data_transform if args.no_aug else data_transform_aug
            )
            
        elif args.poison_type in ["wanet", "adaptive_blend"]:
            cover_train_imgs_path = os.path.join(poison_set_dir, 'cover_imgs.pt')
            cover_train_labels_path = os.path.join(poison_set_dir, 'cover_labels.pt')
            cover_train_set = tools.IMG_Dataset(
                data_dir=cover_train_imgs_path,
                label_path=cover_train_labels_path,
                transforms=data_transform if args.no_aug else data_transform_aug
            )

    else:
        clean_train_set = torch.load(os.path.join(poison_set_dir, 'clean_train_data.pt'))
        backdoor_train_set = torch.load(os.path.join(poison_set_dir, 'backdoor_train_data.pt'))
        backdoor_test_set = torch.load(os.path.join(poison_set_dir, 'backdoor_test_data.pt'))
        
        transform = data_transform if args.no_aug else data_transform_aug
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


        backdoor_test_imgs = []
        backdoor_test_labels = []

        for idx in range(len(backdoor_test_set)):
            sample, target = backdoor_test_set[idx]
            transformed_sample = sample
            backdoor_test_imgs.append(transformed_sample)
            backdoor_test_labels.append(attack_config.target_label[args.dataset])
        
        backdoor_test_imgs = torch.stack(backdoor_test_imgs, dim=0)
        backdoor_test_labels = torch.tensor(backdoor_test_labels)

        clean_train_set = torch.utils.data.TensorDataset(transformed_clean_train_imgs, transformed_clean_train_labels)
        backdoor_train_set = torch.utils.data.TensorDataset(transformed_backdoor_train_imgs, transformed_backdoor_train_labels)
        backdoor_test_set = torch.utils.data.TensorDataset(backdoor_test_imgs, backdoor_test_labels)

    if args.poison_type not in ["wanet", "adaptive_blend"]:
        poisoned_train_set = torch.utils.data.ConcatDataset([clean_train_set, backdoor_train_set])
    else:
        poisoned_train_set = torch.utils.data.ConcatDataset([clean_train_set, cover_train_set, backdoor_train_set])

    return poisoned_train_set, clean_train_set, clean_test_set, backdoor_test_set

def get_checkpoint_path(args):
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

    print(f"Will save to '{checkpoint_path}'.")
    if os.path.exists(checkpoint_path):
        print(f"Model '{checkpoint_path}' already exists! Will override it!")
    if not os.path.exists(checkpoint_path): tools.create_missing_folders(checkpoint_path)
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    tools.setup_seed(args.random_seed)

    img_size, num_classes = supervisor.get_info_dataset(args)
    arch = common_config.arch[args.dataset]
    training_setting = common_config.training_setting

    # set variables in training_setting as global variables for direct usage
    # it includes momentum, weight_decay, epochs, milestones, learning_rate and batch_size
    for key, value in training_setting.items():
        globals()[key] = value
    
    poisoned_train_set, clean_train_set, clean_test_set, backdoor_test_set = prepare_dataset(args)
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    if args.dataset == 'tiny':
        kwargs = {'num_workers': 32, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4, 'pin_memory': True}

    poisoned_train_set_loader = torch.utils.data.DataLoader(
        poisoned_train_set,
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )

    clean_test_set_loader = torch.utils.data.DataLoader(
        clean_test_set,
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )

    if backdoor_test_set is not None:
        backdoor_test_set_loader = torch.utils.data.DataLoader(
            backdoor_test_set,
            batch_size=batch_size, 
            shuffle=False, 
            worker_init_fn=tools.worker_init, 
            **kwargs
        )
        poison_transform = None   
    else:
        poison_transform = supervisor.get_poison_transform(
            poison_type=args.poison_type, 
            dataset_name=args.dataset,
            target_label=attack_config.target_label[args.dataset], 
            trigger_transform=data_transform,
            alpha=args.alpha,
            trigger_name=args.trigger, 
            args=args
        )
    
    checkpoint_save_path = get_checkpoint_path(args)

    model = common_config.arch[args.dataset](num_classes=num_classes)
    # Check if need to resume from the checkpoint
    if os.path.exists(os.path.join(checkpoint_save_path, "meta_info")):
        meta_info = torch.load(os.path.join(checkpoint_save_path, "meta_info"))
    else:
        meta_info = dict()
        meta_info['epoch'] = 0

    if args.resume > 0:
        meta_info['epoch'] = args.resume
        if os.path.exists(checkpoint_save_path):
            model.load_state_dict(torch.load(os.path.join(checkpoint_save_path, 'epoch_'+str(args.resume)+'.pt')))
    elif args.resume_from_meta_info:
        args.resume = meta_info['epoch']
        if os.path.exists(checkpoint_save_path):
            model.load_state_dict(torch.load(os.path.join(checkpoint_save_path, 'epoch_'+str(args.resume)+'.pt')))
    else:
        meta_info['epoch'] = 0


    # model = nn.DataParallel(model)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=momentum, 
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    exe_start_time = datetime.datetime.now()
    training_loss = []
    for epoch in range(1, epochs+1):  # train backdoored base model
        start_time = time.perf_counter()

        # skip to the checkpointed epoch
        if epoch <= args.resume:
            scheduler.step()
            continue

        # train
        model.train()
        preds = []
        labels = []
        for data, target in tqdm(poisoned_train_set_loader):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], elapsed_time))
        scheduler.step()

        # test
        if backdoor_test_set is None:
            ca, asr, loss = tools.test(
                model=model, 
                test_loader=clean_test_set_loader, 
                poison_test=True,
                poison_transform=poison_transform, 
                num_classes=num_classes, 
                args=args,
                return_loss=True
            )
        else:
            ca, _, loss = tools.test(
                model=model, 
                test_loader=clean_test_set_loader, 
                poison_test=False,
                poison_transform=None, 
                num_classes=num_classes, 
                args=args,
                return_loss=True
            )
            asr = tools.test(
                model=model, 
                test_loader=backdoor_test_set_loader, 
                poison_test=False,
                poison_transform=None, 
                num_classes=num_classes, 
                args=args
            )
        meta_info['epoch'] = epoch
        training_loss.append(loss)

        if epoch >= args.checkpoint_save_epoch:
            torch.save(model.state_dict(), checkpoint_save_path + '/epoch_' + str(epoch) + '.pt')
        with open(checkpoint_save_path + "/asr_ca.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\n\t\t")
            file.write("CA:\t\t" + str(ca) + "\n\t\t")
            file.write("ASR:\t" + str(asr) + "\n\n")

    training_loss = torch.tensor(training_loss)
    torch.save(training_loss, os.path.join(checkpoint_save_path, "training_loss"))
    torch.save(meta_info, os.path.join(checkpoint_save_path, "meta_info"))

    # log information
    if args.log:
        log_path = os.path.join('../logs', '%s_random_seed=%d' % (args.dataset, args.random_seed))
        if not os.path.exists(log_path): tools.create_missing_folders(log_path)
        log_path = os.path.join(log_path, '%s' % (exe_start_time.strftime('%Y-%m-%d-%H:%M:%S_')+supervisor.get_info_core(args, include_poison_seed=attack_config.record_poison_seed)))
        
        exe_end_time = datetime.datetime.now()
        with open(log_path+".txt", "a") as file:
            file.write("\n\nExecution Start Time:\n\t"+exe_start_time.strftime('%Y-%m-%d %H:%M:%S'))

            file.write("\n\nArgs:")
            file.write("\n\t\t")
            file.write(json.dumps(args.__dict__))

            file.write("\n\nTraining Dataset:")
            file.write("\n\tcheckpoint_save_path:\t" + checkpoint_save_path)
            

            file.write("\n\nTraining Setting:")
            file.write("\n\tarch:\t" + arch.__name__)
            file.write("\n\tmomentum:\t" + str(momentum))
            file.write("\n\tweight_decay:\t" + str(weight_decay))
            file.write("\n\tepochs:\t" + str(epochs))
            file.write("\n\tmilestones:\t" + str(milestones))
            file.write("\n\tlearning_rate:\t" + str(learning_rate))
            file.write("\n\tbatch_size:\t" + str(batch_size))

            file.write("\n\tdata_transform:\n\t\t" + str(data_transform))
            file.write("\n\tdata_transform_aug:\n\t\t" + str(data_transform_aug))

            file.write("\n\nExecution End Time:\n\t"+ str(exe_end_time))
            file.write("\n\nExecution Total Time:\n\t"+ str(exe_end_time-exe_start_time))
