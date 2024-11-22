"""
config of attacks
"""

poison_seed = 0             # random seed for creating suspicous training set
record_poison_seed = False  # set to False if the data is from https://github.com/SCLBD/BackdoorBench


target_label = {    # default target class is 0 (without loss of generality)
    'cifar10' : 0,
    'gtsrb' : 0,
    'tiny': 0,
}

triggers_dir = '../triggers'

trigger_default = {     # trigger of attacks
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