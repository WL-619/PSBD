"""
Code modified from https://github.com/vtu81/backdoor-toolbox/blob/main/poison_tool_box/adaptive_blend.py
"""

import torch
import random
from math import sqrt


def issquare(x):
    tmp = sqrt(x)
    tmp2 = round(tmp)
    return abs(tmp - tmp2) <= 1e-8


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask


class poison_generator():

    def __init__(self, img_size, dataset, poisoning_ratio, trigger, target_label, alpha=0.2, cover_rate=0.01,
                 pieces=16, mask_rate=0.5):

        self.img_size = img_size
        self.dataset = dataset
        self.poisoning_ratio = poisoning_ratio
        self.target_label = target_label  # by default : target_class = 0
        self.trigger = trigger
        self.alpha = alpha
        self.cover_rate = cover_rate
        assert abs(round(sqrt(pieces)) - sqrt(pieces)) <= 1e-8
        assert img_size % round(sqrt(pieces)) == 0
        self.pieces = pieces
        self.mask_rate = mask_rate
        self.masked_pieces = round(self.mask_rate * self.pieces)

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poisoning_ratio)
        backdoor_indices = id_set[:num_poison]

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        
        clean_indices = id_set[num_poison + num_cover:]

        backdoor_indices.sort()  # increasing order
        cover_indices.sort()
        clean_indices.sort()
        print('backdoor samples num: ', len(backdoor_indices))


        clean_img_set = []
        cover_img_set = []
        backdoor_img_set = []

        clean_label_set = []
        cover_label_set = []
        backdoor_label_set = []

        mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)

        for i in backdoor_indices:
            img, gt = self.dataset[i]
            img = img + self.alpha * mask * (self.trigger - img)
            gt = self.target_label
            backdoor_img_set.append(img)
            backdoor_label_set.append(gt)

        for i in cover_indices:
            img, gt = self.dataset[i]
            img = img + self.alpha * mask * (self.trigger - img)
            cover_img_set.append(img)
            cover_label_set.append(gt)
            
        for i in clean_indices:
            img, gt = self.dataset[i]
            clean_img_set.append(img)
            clean_label_set.append(gt)

        clean_label_set = torch.LongTensor(clean_label_set)
        cover_label_set = torch.LongTensor(cover_label_set)
        backdoor_label_set = torch.LongTensor(backdoor_label_set)

        return clean_img_set, clean_label_set, backdoor_img_set, backdoor_label_set, cover_img_set, cover_label_set


class poison_transform():

    def __init__(self, img_size, trigger, target_label, alpha=0.15):
        self.img_size = img_size
        self.target_label = target_label
        self.trigger = trigger
        self.alpha = alpha
        self.mask = get_trigger_mask(img_size, 16, 8)

    def transform(self, img, label, return_mask=False):
        img, label = img.clone(), label.clone()
        mask = self.mask.to(img.device)
        trigger = self.trigger.to(img.device)
        img = img + self.alpha * (trigger - img)
        label[:] = self.target_label
        masked_trigger = mask * trigger
        
        if return_mask:
            return img, label, masked_trigger
        return img, label