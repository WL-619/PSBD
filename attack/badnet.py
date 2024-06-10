"""
Code modified from https://github.com/vtu81/backdoor-toolbox/blob/main/poison_tool_box/badnet.py
"""

import torch
import random

class poison_generator():

    def __init__(self, img_size, dataset, poisoning_ratio, trigger_mark, trigger_mask, target_label, alpha=1.0):
        self.img_size = img_size
        self.dataset = dataset
        self.poisoning_ratio = poisoning_ratio
        self.target_label = target_label
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poisoning_ratio)
        backdoor_indices = id_set[:num_poison]
        clean_indices = id_set[num_poison:]

        backdoor_indices.sort() # increasing order
        clean_indices.sort()

        print('backdoor samples num: ', len(backdoor_indices))

        clean_img_set = []
        backdoor_img_set = []

        clean_label_set = []
        backdoor_label_set = []

        backdoor_maker = poison_transform(
            self.img_size, 
            self.trigger_mark, 
            self.trigger_mask,
            self.target_label,
            self.alpha
        )
        for i in backdoor_indices:
            img, gt = self.dataset[i]
            img, gt = backdoor_maker.transform(img, torch.tensor(gt).unsqueeze(0))
            backdoor_img_set.append(img)
            backdoor_label_set.append(gt.item())

        for i in clean_indices:
            img, gt = self.dataset[i]
            clean_img_set.append(img)
            clean_label_set.append(gt)
        
        clean_label_set = torch.LongTensor(clean_label_set)
        backdoor_label_set = torch.LongTensor(backdoor_label_set)

        return clean_img_set, clean_label_set, backdoor_img_set, backdoor_label_set



class poison_transform():   # make poison samples
    def __init__(self, img_size, trigger_mark, trigger_mask, target_label, alpha=1.0):
        self.img_size = img_size
        self.target_label = target_label
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

    def transform(self, imgs, labels):
        imgs, labels = imgs.clone(), labels.clone()
        imgs = imgs + self.alpha * self.trigger_mask.to(imgs.device) * (self.trigger_mark.to(imgs.device) - imgs)
        labels[:] = self.target_label

        return imgs, labels