"""
Please clone the repo https://github.com/HanxunH/CognitiveDistillation first
"""

from tqdm import tqdm
from utils import tools, supervisor
import numpy as np
import torch
import sys
sys.path.append('../../')
from CognitiveDistillation import detection, analysis
import os

class CDL():
    def __init__(
            self, 
            args,
            p=1, 
            gamma=0.001,
            beta = 1.0,
            num_steps = 100,
            step_size = 0.1,
            mask_channel = 1,
            norm_only = False,
        ):
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)
        self.args = args
        hyper_params = (p, mask_channel, gamma, beta, num_steps, step_size)
        file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}_'.format(*hyper_params)
        file_extension = file_extension + f"model_{args.select_model}.pt"
        train_filename = 'cd_train_mask_' + file_extension
        
        self.save_path = os.path.join("./results", "cd", args.dataset, str(args.poisoning_ratio), args.poison_type, str(args.random_seed))
        self.filename = os.path.join(self.save_path, train_filename)  
        self.detector = detection.CognitiveDistillation(
            p=p,
            gamma=gamma,
            beta=beta,
            num_steps=num_steps,
            lr=step_size,
            mask_channel=mask_channel,
            norm_only=norm_only
        )
        self.analysis = analysis.CognitiveDistillationAnalysis(od_type='CD', norm_only=norm_only)
    
    def cleanse(self, inspection_split_loader, clean_set_loader, selcected_model):
        selcected_model.eval()
        train_results = []
        for images, labels in tqdm(inspection_split_loader):
            images, labels = images.cuda(), labels.cuda()
            batch_rs = self.detector(selcected_model, images, labels)
            train_results.append(batch_rs.detach().cpu())
        train_results = torch.cat(train_results, dim=0)

        val_results = []
        for images, labels in tqdm(clean_set_loader):
            images, labels = images.cuda(), labels.cuda()
            batch_rs = self.detector(selcected_model, images, labels)
            val_results.append(batch_rs.detach().cpu())
        val_results = torch.cat(val_results, dim=0)
        
        
        if not os.path.exists(self.save_path):
            tools.create_missing_folders(self.save_path)
        torch.save(train_results, self.filename)
        
        self.analysis.train(val_results)
        train_scores = self.analysis.predict(train_results, t=1)
        return train_scores
        

def cleanser(inspection_set, selcected_model, args, clean_set):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # The dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
            inspection_set,
            batch_size=128, 
            shuffle=False, 
            **kwargs
        )

    # A small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
            clean_set,
            batch_size=128, 
            shuffle=True, 
            **kwargs
        )
    
    worker = CDL(args)
    suspicious_indices = worker.cleanse(inspection_split_loader, clean_set_loader, selcected_model)

    return suspicious_indices
