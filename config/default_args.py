"""
Configuration file for default arguments and parser choices
"""

parser_choices = {
    'dataset': ['cifar10', 'gtsrb', 'tiny'],
    'poison_type':
        ['badnet', 'blend', 'trojannn', 'lc', 'wanet', 'issba', 'adaptive_blend'],
    
    # Proportion of backdoor data in the entire training dataset
    'poisoning_ratio': [i / 1000.0 for i in range(0, 500)],
    
    # The ratio of data samples with backdoor trigger but without label modification to target class
    'cover_rate': [i / 1000.0 for i in range(0, 500)], 
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poisoning_ratio': 0.1,
    'cover_rate': 0,
    'alpha': 1.0,   # Blend strength for blend or adaptive_blend attack
}

random_seed = 42    # Random seed for model training
checkpoint_save_epoch = 1  # Save model if current epoch >= checkpoint_save_epoch
checkpoint_save_path = "../checkpoints"
fig_save_path = "../figs"