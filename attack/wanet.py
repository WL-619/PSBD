"""
Code modified from https://github.com/vtu81/backdoor-toolbox/blob/main/poison_tool_box/WaNet.py
"""

import torch
import random
import torch.nn.functional as F

class poison_generator():

    def __init__(self, img_size, dataset, poisoning_ratio, cover_rate, target_label, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1):
        """
        Initializes the generator for poisoned samples in the wanet backdoor attack
        
        Args:
            img_size: Input image size (assumed square)
            dataset: The training dataset
            poisoning_ratio: The ratio of poisoned samples
            cover_rate: The ratio of cover samples that contain the trigger but retain their original labels
            target_label: Target class label
            identity_grid: Identity grid
            noise_grid: Noise grid
        """
        
        self.img_size = img_size
        self.dataset = dataset
        self.poisoning_ratio = poisoning_ratio
        self.cover_rate = cover_rate

        self.target_label = target_label

        # The number of images
        self.num_img = len(dataset)

        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid

        self.grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        self.grid_temps = torch.clamp(self.grid_temps, -1, 1)

    def generate_poisoned_training_set(self):
        # Random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poisoning_ratio)
        backdoor_indices = id_set[:num_poison]

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison+num_cover]

        clean_indices = id_set[num_poison+num_cover:]

        backdoor_indices.sort() # Increasing order
        cover_indices.sort()
        clean_indices.sort()

        print(f'backdoor samples num: {len(backdoor_indices)}, cover samples num: {len(cover_indices)}.')

        clean_img_set = []
        cover_img_set = []
        backdoor_img_set = []

        clean_label_set = []
        cover_label_set = []
        backdoor_label_set = []

        ins = torch.rand(1, self.img_size, self.img_size, 2) * 2 - 1
        grid_temps2 = self.grid_temps + ins / self.img_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        for i in backdoor_indices:
            img, gt = self.dataset[i]
            img = F.grid_sample(img.unsqueeze(0), self.grid_temps, align_corners=True)[0]
            gt = self.target_label
            backdoor_img_set.append(img)
            backdoor_label_set.append(gt)

        for i in cover_indices:
            img, gt = self.dataset[i]
            img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
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



class poison_transform():   # Make poison samples
    def __init__(self, img_size, normalizer, denormalizer, identity_grid, noise_grid, target_label, s=0.5, k=4, grid_rescale=1,):

        self.img_size = img_size
        self.target_label = target_label
        self.normalizer = normalizer
        self.denormalizer = denormalizer

        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid.cuda()
        self.noise_grid = noise_grid.cuda()

        self.grid_temps = (identity_grid + s * noise_grid / self.img_size) * self.grid_rescale
        self.grid_temps = torch.clamp(self.grid_temps, -1, 1)

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = self.denormalizer(data)
        data = F.grid_sample(data, self.grid_temps.to(data.device).repeat(data.shape[0], 1, 1, 1), align_corners=True)
        data = self.normalizer(data)
        labels[:] = self.target_label

        return data, labels