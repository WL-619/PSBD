"""
Code modified from https://github.com/vtu81/backdoor-toolbox/blob/main/other_defenses_tool_box/scale_up.py
"""

from tqdm import tqdm
from utils import tools, supervisor
import numpy as np
import torch


class ScaleUp():
    def __init__(self, args, model, clean_set_loader, scale_set=None, threshold=None, with_clean_data=True):
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)

        if scale_set is None:
            scale_set = [3, 5, 7, 9, 11]
        if threshold is None:
            self.threshold = 0.5
        self.scale_set = scale_set
        self.args = args

        self.with_clean_data = with_clean_data

        self.mean = None
        self.std = None
        
        self.model = model
        self.init_spc_norm(clean_set_loader)        
    
    def cleanse(self, inspection_split_loader, inspect_correct_predition_only=False):
        self.model.eval()
        total_spc = []
        for idx, (imgs, labels) in enumerate(tqdm(inspection_split_loader)):
            imgs = imgs.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(imgs) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            
            pred = torch.argmax(self.model(imgs), dim=1) # model prediction
            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == pred
            spc /= len(self.scale_set)
            
            if self.with_clean_data:
                spc = (spc - self.mean) / self.std
                
            total_spc.append(spc)
        
        total_spc = torch.cat(total_spc, dim=0)
        y_score = total_spc.cpu().detach()
        y_pred = (y_score >= self.threshold).cpu().detach()
        
        
        if inspect_correct_predition_only:
            # Only consider:
            #   1) clean inputs that are correctly predicted
            #   2) poison inputs that successfully trigger the backdoor
            clean_pred_correct_mask = []
            poison_source_mask = []
            poison_attack_success_mask = []
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
                # on poison data
                data, target = data.cuda(), target.cuda()
                
                
                clean_output = self.model(data)
                clean_pred = clean_output.argmax(dim=1)
                mask = torch.eq(clean_pred, target) # only look at those samples that successfully attack the DNN
                clean_pred_correct_mask.append(mask)
                
                
                poison_data, poison_target = self.poison_transform.transform(data, target)
                
                if args.poison_type == 'TaCT':
                    mask = torch.eq(target, config.source_class)
                else:
                    # remove backdoor data whose original class == target class
                    mask = torch.not_equal(target, poison_target)
                poison_source_mask.append(mask.clone())
                
                poison_output = self.model(poison_data)
                poison_pred = poison_output.argmax(dim=1)
                mask = torch.logical_and(torch.eq(poison_pred, poison_target), mask) # only look at those samples that successfully attack the DNN
                poison_attack_success_mask.append(mask)

            clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
            poison_source_mask = torch.cat(poison_source_mask, dim=0)
            poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
            
            preds_clean = y_pred[:int(len(y_pred) / 2)]
            preds_poison = y_pred[int(len(y_pred) / 2):]
            # print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(preds_clean)].sum(), len(clean_pred_correct_mask),
            #                                         clean_pred_correct_mask[torch.logical_not(preds_clean)].sum() / len(clean_pred_correct_mask)))
            # print("ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(preds_poison)].sum(), poison_source_mask.sum(),
            #                              poison_attack_success_mask[torch.logical_not(preds_poison)].sum() / poison_source_mask.sum() if poison_source_mask.sum() > 0 else 0))
        
            mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)
            y_pred = y_pred[mask.cpu()]
            y_score = y_score[mask.cpu()]

        return y_pred
        

    def init_spc_norm(self, clean_set_loader):
        total_spc = []
        self.model.eval()
        for idx, (clean_img, labels) in enumerate(clean_set_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)
        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()

def cleanser(inspection_set, selcected_model, args, clean_set, inspect_correct_predition_only=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
            inspection_set,
            batch_size=128, 
            shuffle=False, 
            **kwargs
        )

    # a small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
            clean_set,
            batch_size=128, 
            shuffle=True, 
            **kwargs
        )
    
    worker = ScaleUp(args, selcected_model, clean_set_loader)
    suspicious_indices = worker.cleanse(inspection_split_loader, inspect_correct_predition_only)

    return suspicious_indices