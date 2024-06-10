'''
Modified from https://github.com/vtu81/backdoor-toolbox/blob/main/utils/tools.py
'''
import torch, torchvision
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np
from torchvision.utils import save_image
from utils import supervisor
from tqdm import tqdm

class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None):
        """
            Args:
                data_dir: directory of the data

                label_path: path to data labels

                transforms: image transformation to be applied
                
        """
        self.dir = data_dir
        self.img_set = None
        if 'data' not in self.dir: # if new version
            self.img_set = torch.load(data_dir)
        self.gt = torch.load(label_path)    # ground truth, i.e. data label
        self.transforms = transforms
        if 'data' not in self.dir: # if new version, remove ToTensor() from the transform list
            self.transforms = []
            for t in transforms.transforms:
                if not isinstance(t, torchvision.transforms.ToTensor):
                    self.transforms.append(t)
            self.transforms = torchvision.transforms.Compose(self.transforms)

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label

        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        
        if self.img_set is not None: # if new version
            img = self.img_set[idx]
        else: # if old version
            img = Image.open(os.path.join(self.dir, '%d.png' % idx))
        
        if self.transforms is not None:
            img = self.transforms(img)

        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label+1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label


# test CA and ASR
def test(model, test_loader, args, poison_test = False, poison_transform=None, return_loss=False, num_classes=10, enable_drop=False, drop_p=0):
    model.eval()
    if enable_drop:
        enable_dropout(model, drop_p)
    clean_correct = 0
    poison_correct = 0
    tot = 0
    num_non_target_label = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    img_size, num_classes = supervisor.get_info_dataset(args)

    class_dist = np.zeros((num_classes))

    with torch.no_grad():
        for imgs, target in tqdm(test_loader):

            imgs, target = imgs.cuda(), target.cuda()
            clean_output = model(imgs)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size


            for bid in range(this_batch_size):
                if clean_pred[bid] == target[bid]:
                    class_dist[target[bid]] += 1

            if poison_test:
                clean_target = target
                imgs, target = poison_transform.transform(imgs, target)

                poison_output = model(imgs)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                target_label = target[0].item()
                for bid in range(this_batch_size):
                    if clean_target[bid]!=target_label:
                        num_non_target_label+=1
                        if poison_pred[bid] == target_label:    # attack success
                            poison_correct+=1
    
    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
            clean_correct, tot, clean_correct/tot,
            tot_loss/tot
    ))
    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_label, poison_correct / num_non_target_label))
    
    if poison_test:
        if return_loss:
            return clean_correct/tot, poison_correct / num_non_target_label, tot_loss/tot
        return clean_correct/tot, poison_correct / num_non_target_label
    if return_loss:
        return clean_correct/tot, None, tot_loss/tot
    return clean_correct/tot, None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_dataset(dataset, path):
    num = len(dataset)
    label_set = []

    if not os.path.exists(path):
        os.mkdir(path)

    img_dir = os.path.join(path,'imgs')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


    for i in range(num):
        img, gt = dataset[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Test Set] Save %s' % img_file_path)
        label_set.append(gt)

    label_set = torch.LongTensor(label_set)
    label_path = os.path.join(path, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Test Set] Save %s' % label_path)

def enable_dropout(model, drop_p):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = drop_p

def create_missing_folders(path):
    folders = path.split(os.sep)

    for i in range(0, len(folders)):
        folder = os.path.join(*folders[:i+1])
        if not os.path.exists(folder):
            os.mkdir(folder)