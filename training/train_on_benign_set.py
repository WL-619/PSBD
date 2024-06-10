import argparse
import os, sys
sys.path.append('../')
import time
from tqdm import tqdm
from config import attack_config, common_config, default_args
from utils import tools, supervisor
import json
from torchvision import datasets
from torch import nn
import torch
import datetime
import time

def add_arguments(parser):
    """
        Args:
            dataset (str): The dataset we used
            
            resume (int): Resume from the checkpoint

            resume_from_meta_info (bool): Resume from the meta info

            no_aug (bool, default=False): Whether to use data augmentation. If True, data augmentation will be applied

            no_normalize (bool, default=False): Whether to use data normalization. If True, data normalization will be applied

            log (bool, default=False): Whether to save log file

            random_seed (int): Random seed for model training, rather than the seed used for generating the poisoned dataset

            checkpoint_save_path (str): Path to save checkpoints for loading checkpoints

            checkpoint_save_epoch (int, default=1): Save model if current epoch >= checkpoint_save_epoch

    """

    parser.add_argument('-dataset', type=str, required=False,
                        default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'])
    parser.add_argument('-resume', type=int, required=False, default=0)
    parser.add_argument('-resume_from_meta_info', default=False, action='store_true')
    parser.add_argument('-no_aug', default=False, action='store_true')
    parser.add_argument('-no_normalize', default=False, action='store_true')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-random_seed', type=int, required=False, default=default_args.random_seed)
    parser.add_argument('-checkpoint_save_path', type=str, required=False, default=default_args.checkpoint_save_path)
    parser.add_argument('-checkpoint_save_epoch', type=int, required=False, default=default_args.checkpoint_save_epoch)
    return parser

def prepare_dataset(args):
    clean_data_dir = common_config.clean_data_dir
    attack_config.record_poison_seed = True
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(
            os.path.join(clean_data_dir, 'cifar10'), 
            train=True,
            download=True, 
            transform=data_transform if args.no_aug else data_transform_aug
        )

    elif args.dataset == 'tiny':
        train_set = datasets.ImageFolder(
            os.path.join(os.path.join(clean_data_dir, 'tiny'), 'train'),
            data_transform if args.no_aug else data_transform_aug
        )
    else:
        raise NotImplementedError('Dataset is not implemented')

    test_dir = os.path.join(clean_data_dir, args.dataset)
    if args.dataset == 'cifar10':
        test_set = datasets.CIFAR10(
            root=test_dir, 
            transform=data_transform, 
            train=False
        )
    elif args.dataset == 'tiny':
        test_dir = os.path.join(test_dir, 'val')
        test_set = datasets.ImageFolder(
            root=test_dir, 
            transform=data_transform
        )
    return train_set, test_set

def get_checkpoint_path(args):
    if args.no_aug:
        prefix = 'no_aug_'
    else:
        prefix = 'has_aug_'

    if args.no_normalize:
        prefix = prefix + 'no_norm'
    else:
        prefix = prefix + 'has_norm'

    checkpoint_save_path = os.path.join(args.checkpoint_save_path, arch.__name__, args.dataset, prefix, 'clean_model', 'random_seed_'+str(args.random_seed))

    print(f"Will save to '{checkpoint_save_path}'.")
    if os.path.exists(checkpoint_save_path):
        print(f"Model '{checkpoint_save_path}' already exists! Will override it!")
    else:
        tools.create_missing_folders(checkpoint_save_path)

    return checkpoint_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    tools.setup_seed(args.random_seed)
    exe_start_time = datetime.datetime.now()

    img_size, num_classes = supervisor.get_info_dataset(args)
    arch = common_config.arch[args.dataset]
    training_setting = common_config.training_setting

    # set variables in training_setting as global variables for direct usage
    # it includes momentum, weight_decay, epochs, milestones, learning_rate and batch_size
    for key, value in training_setting.items():
        globals()[key] = value

    train_set, test_set = prepare_dataset(args)

    if args.dataset == 'tiny':
        kwargs = {'num_workers': 32, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4, 'pin_memory': True}


    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=tools.worker_init, 
        **kwargs
    )
    
    checkpoint_save_path = get_checkpoint_path(args)
    model = common_config.arch[args.dataset](num_classes)
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
        for data, target in tqdm(train_set_loader):
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('Benign Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], elapsed_time))
        scheduler.step()

        # test
        ca, asr, loss = tools.test(
            model=model, 
            test_loader=test_set_loader, 
            poison_test=False,
            poison_transform=None, 
            num_classes=num_classes, 
            args=args,
            return_loss=True
        )
        meta_info['epoch'] = epoch
        training_loss.append(loss)

        if epoch >= args.checkpoint_save_epoch:
            torch.save(model.state_dict(), checkpoint_save_path + '/epoch_' + str(epoch) + '.pt')
        with open(checkpoint_save_path + "/ca.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\n\t\t")
            file.write("CA:\t\t" + str(ca) + "\n\n")


    training_loss = torch.tensor(training_loss)
    torch.save(training_loss, os.path.join(checkpoint_save_path, "training_loss"))
    torch.save(meta_info, os.path.join(checkpoint_save_path, "meta_info"))

    # log information
    if args.log:
        log_path = os.path.join('../logs', '%s_random_seed=%d' % (args.dataset, args.random_seed))
        if not os.path.exists(log_path): tools.create_missing_folders(log_path)
        log_path = os.path.join(log_path, '%s' % (exe_start_time.strftime('%Y-%m-%d-%H:%M:%S')+'benign model'))
        
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

            file.write("\n\nExecution End Time:\n\t"+ str(exe_end_time))
            file.write("\n\nExecution Total Time:\n\t"+ str(exe_end_time-exe_start_time))
