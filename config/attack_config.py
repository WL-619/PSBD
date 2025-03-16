"""
Configuration file for backdoor attack settings
"""

# Random seed used for creating a suspicious training dataset
poison_seed = 0

# Indicates whether to record the poison seed
# If the dataset is sourced from https://github.com/SCLBD/BackdoorBench, set this to False
record_poison_seed = False

# By default, the target class label is set to 0 for all datasets
target_label = {
    'cifar10' : 0,
    'gtsrb' : 0,
    'tiny': 0,
}

triggers_dir = '../triggers'

trigger_default = {     # Trigger of attacks
    'cifar10': {
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'trojannn': 'none',
        'lc' : 'badnet_patch4_32.png',
        'wanet': 'none',
        'issba': 'none',
        'adaptive_blend': 'hellokitty_32.png',
    },
    'gtsrb': {
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'trojannn': 'none',
        'lc' : 'badnet_patch4_32.png',
        'wanet': 'none',
        'issba': 'none',
        'adaptive_blend': 'hellokitty_32.png',
    },

    'tiny': {
        'badnet': 'badnet_patch_64.png',
        'blend' : 'hellokitty_64.png',
        'trojannn': 'none',
        'lc': 'badnet_patch4_64.png',
        'wanet': 'none',
        'issba': 'none',
        'adaptive_blend': 'random_64.png',
    }
}