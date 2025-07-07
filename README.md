# [PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection (CVPR 2025)](https://github.com/WL-619/PSBD)

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2406.05826-b31b1b.svg)](https://arxiv.org/abs/2406.05826)&nbsp;
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

</div>

## 🔍 Framework Overview
✨ We propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. 

💡 The introduction of dropout during the inference stage induces a neuron bias effect in the model, causing the final feature maps of clean data and backdoor data to become highly similar, ultimately leading to the occurrence of the Prediction Shift phenomenon, which serves as a basis for detecting backdoor training data.

![PSBD Pipeline](./images/PSBD.png)

## ⚙️ Environments
```bash
conda create -n PSBD python=3.11
conda activate PSBD

git clone https://github.com/WL-619/PSBD
cd PSBD
pip3 install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Data Download
- The original CIFAR-10 dataset will be automatically downloaded.

- To download and process the Tiny ImageNet dataset, you can use the script we provided by running `bash tiny_download_process.sh`. It will automatically save the dataset to the appropriate directory.

- We provide data for TrojanNN and ISSBA attacks on CIFAR-10 and Tiny ImageNet datasets, as well as Label-Consistent attack on the Tiny ImageNet dataset. All these data are sourced from the [BackdoorBench](https://github.com/SCLBD/BackdoorBench) repository, and you can download them from [OneDrive](https://1drv.ms/f/s!Ajixv2f3vMZfgQxLlGK26T5r5Qa1?e=3ldNAS).

The directory structure of the downloaded `poisoned_train_set` folder should be:

```
poisoned_train_set
├── cifar10
│   ├── cifar10_issba_0_1.zip
│   ├── cifar10_issba_0_05.zip
│   ├── cifar10_trojannn_0_1.zip
│   ├── cifar10_trojannn_0_05.zip
├── tiny
│   ├──tiny_issba_0_1.zip
│   ├──tiny_issba_0_05.zip
│   ├──tiny_trojannn_0_1.zip
│   ├──tiny_trojannn_0_05.zip
│   ├──tiny_lc_backdoor_train.zip
│   ├──tiny_lc_backdoor_test.zip
```

Please run the command `bash process_bench_data.sh` to process the downloaded data. The data will be automatically placed in the appropriate directories.
```bash
cd poisoned_train_set
bash process_bench_data.sh
```

### Step 2: Create Clean Validation Data
Please initialize the label-free clean extra validation data using the following command before starting any experiments:
```bash
cd extra_val_set

python create_extra_val_set.py -dataset cifar10
python create_extra_val_set.py -dataset tiny
```

### Step 3: Create Poisoned Training Data
Before launching `lc` attack on cifar10, run [poisoned_train_set/lc_cifar10.sh](/poisoned_train_set/lc_cifar10.sh).

Some examples for creating poisoned training data:
```bash
cd poisoned_train_set

python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 

python create_poisoned_set.py -dataset cifar10 -poison_type lc -poisoning_ratio 0.1
python create_lc_on_tiny.py -adv_data_path './tiny/lc_backdoor_train/tiny_lc_train.npy' -poisoning_ratio 0.1

# Data from bench
python create_poisoned_set_from_bench.py -dataset cifar10 -poison_type trojannn -data_path './cifar10/trojannn_0_1/' -poisoning_ratio 0.1
```

### Step 4: Train on Poisoned Data
After creating the poisoned training data, you can train the backdoor model with the following command:
```bash
cd training

python train_on_poison_set.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize

# If the data from bench, please add -load_bench_data parameter
python train_on_poison_set.py -dataset cifar10 -poison_type trojannn -poisoning_ratio 0.1 -no_aug -no_normalize -load_bench_data

# Data augmentation during the model training was employed exclusively for Adaptive-Blend on CIFAR-10 and all experiments on Tiny ImageNet 
# to achieve an attack success rate exceeding 85%
python train_on_poison_set.py -dataset cifar10 -poison_type adaptive_blend -poisoning_ratio 0.01 -cover_rate 0.01 -alpha 0.2 -test_alpha 0.25 -no_normalize

python train_on_poison_set.py -dataset tiny -poison_type trojannn -poisoning_ratio 0.1 -no_normalize -load_bench_data
```

### Step 5: Detect Backdoor Data
Once you have the trained backdoor model, you can utilize our PSBD and other detection methods to detect the backdoor data in the training dataset using the following command:
```bash
cd detection

# Our PSBD method
python psbd.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize

# Baseline methods: scan, spectre, spectral_signature(ss), strip, scp, cdl
python baseline_metric.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize -baseline ss
```

## 🔑 Pilot Studies
To view the results of the pilot studies in our paper, you can run the following command. Please ensure you have the trained models before conducting the pilot studies.
```bash
cd pilot_study

# Pilot Study 1: MC-Dropout uncertainty (refer to our main paper for details)
python pilot.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize

# Pilot Study 2: MC-Dropout uncertainty combined with input uncertainty (scaling up the image pixel values), as described in the appendix of our paper
python pilot.py -dataset cifar10 -poison_type badnet -poisoning_ratio 0.1 -no_aug -no_normalize -scale 3
```
## 🕹️ Train on Benign Data
You can also train the model on the benign dataset to exmaine the performance of clean model.
```bash
python train_on_benign_set.py -dataset cifar10 -no_aug -no_normalize

python train_on_benign_set.py -dataset tiny -no_normalize
```

## 📄 Citation
If you find our work to be useful for your research, please cite
```
@inproceedings{li2025psbd,
  title={PSBD: Prediction shift uncertainty unlocks backdoor detection},
  author={Li, Wei and Chen, Pin-Yu and Liu, Sijia and Wang, Ren},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10255--10264},
  year={2025}
}
```

## 🙏 Acknowledgement
This repo benefits from [backdoor-toolbox](https://github.com/vtu81/backdoor-toolbox/), [BackdoorBench](https://github.com/SCLBD/BackdoorBench). Thanks for their wonderful works.
