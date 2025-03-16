"""
Code modified from https://github.com/vtu81/backdoor-toolbox/blob/main/poison_tool_box/clean_label.py
"""

import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, adv_imgs, poisoning_ratio, trigger, mask, target_label):
        """
        Initializes the generator for poisoned samples in the badnet backdoor attack
        
        Args:
            img_size: Input image size (assumed square)
            dataset: The training dataset
            adv_imgs: The adversarial images
            poisoning_ratio: The ratio of poisoned samples
            trigger: The backdoor trigger pattern tensor
            mask: The trigger mask tensor
            target_label: Target class label
        """

        self.img_size = img_size
        self.dataset = dataset
        self.adv_imgs = adv_imgs
        self.poisoning_ratio = poisoning_ratio
        self.trigger = trigger
        self.mask = mask
        self.target_label = target_label # By default : target_label = 0

        # The shape of the patch trigger
        self.dx, self.dy = mask.shape

        # The number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        # Random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poisoning_ratio)
        backdoor_indices = id_set[:num_poison]
        clean_indices = id_set[num_poison:]

        backdoor_indices.sort() # Increasing order
        clean_indices.sort()

        print('backdoor samples num: ', len(backdoor_indices))

        clean_img_set = []
        backdoor_img_set = []

        clean_label_set = []
        backdoor_label_set = []

        backdoor_maker = poison_transform(
            self.img_size, 
            self.trigger, 
            self.mask,
            self.target_label,
        )
        for i in backdoor_indices:
            _, gt = self.dataset[i]
            img = self.adv_imgs[i]
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




class poison_transform():
    def __init__(self, img_size, trigger, mask, target_label=0):
        self.img_size = img_size
        self.target_label = target_label # By default : target_label = 0
        self.trigger = trigger
        self.mask = mask
        self.dx, self.dy = mask.shape

    def transform(self, imgs, labels):

        imgs = imgs.clone()
        labels = labels.clone()

        # Transform clean samples to poison samples
        labels[:] = self.target_label
        imgs = imgs + self.mask.to(imgs.device) * (self.trigger.to(imgs.device) - imgs)

        return imgs, labels