import argparse
import os
import sys
import warnings

from utils.load_config import get_args_to_config
from train_model import train_model
from evaluate_model import test_model, test_pretrain_model
from pretrain_model import pretrain_model


# Press the green button in the gutter to run the script.1
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # arguments
    parser = argparse.ArgumentParser('Interface for the task')
    parser.add_argument('--task', type=str, default='reg_finetune', choices=['rrl_pretrain','reg_finetune', 'cls_finetune'])
    parser.add_argument('--model', type=str, default='MLP_TTE', choices=['MLP_TTE','MLP_CLS', 'SimPRL'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'resume', 'test'])
    parser.add_argument('--dataset', type=str, default='chengdu')
    parser.add_argument('--identify', type=str, default='any_text_description')
    parser.add_argument('--gpu', type=int, default=0)

    # get arguments
    config = get_args_to_config(base_dir = os.path.dirname(os.path.abspath(__file__)), parser=parser, is_evaluation=False)

    print(f"{config['mode']} {config['task']} load {config['model']} on {config['dataset']}. identify: {config['identify']}")

    if 'pretrain' in config['task']:
        if config['mode'] == 'test':
            test_pretrain_model(config)
        else:
            pretrain_model(config)
    elif 'finetune' in config['task'] or 'end2end' in config['task']:
        if config['mode'] == 'test':
            test_model(config)
        else:
            train_model(config)

    sys.exit()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
