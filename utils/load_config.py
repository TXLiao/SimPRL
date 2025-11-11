import argparse
import json
import os
import sys
import torch


def get_args_to_config(base_dir, parser, is_evaluation: bool = False):
    try:
        add_other_args(parser)
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        dict_args = vars(args)
        other_args = {key: val
                      for key, val in dict_args.items()
                      if key not in ['task', 'model', 'dataset', 'mode', 'dataset', 'identify', 'gpu'] and val is not None}
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit()

    config = ConfigParser(base_dir, args.task, args.model, args.dataset, args.mode, args.identify, args.gpu, other_args)

    return config

def load_TTE_best_configs(args: argparse.Namespace):
    # model specific settings
    if args.model_name == '':
        pass
    pass

def add_other_args(parser):
    data = other_arguments
    for arg in data:
        if data[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of str':
            parser.add_argument('--{}'.format(arg), nargs='+', type=str,
                                default=data[arg]['default'], help=data[arg]['help'])

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')

def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x

class ConfigParser(object):

    def __init__(self,base_dir, task, model, dataset, mode, identify, gpu, other_args=None):
        self.config = {}
        self._parse_basic_config(base_dir, task, model, dataset, mode, identify, gpu, other_args, hyper_config_dict=None)
        self._load_config()
        # self._init_device()

    def _parse_basic_config(self,base_dir, task, model, dataset, mode, identify, gpu, other_args=None, hyper_config_dict=None):
        self.config['base_dir'] = base_dir
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['identify'] = identify
        self.config['mode'] = mode
        self.config['gpu'] = gpu
        if other_args is not None:
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_config(self):
        config_dir = os.path.join(self.config['base_dir'],'config')

        with open(f'{config_dir}/configs.json', 'r') as f:
            task_config = json.load(f)
            task_config = task_config[self.config['task']]
            model = self.config['model']
            self.config['dataset_config'] = f'dataset/' + self.config['dataset'] + '/' + task_config[model]['dataset_config']
            self.config['executor_config'] = f'executor/' + task_config[model]['executor_config']
            self.config['model_config'] = f'model/' + task_config[model]['model_config']

        for file_name in [self.config['dataset_config'],
                          self.config['executor_config'], self.config['model_config']]:
            with open(f'{config_dir}/{file_name}.json', 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        # if self.config['backbone'] is not None:
        #     with open(f"{config_dir}/model/{self.config['backbone']}.json", 'r') as f:
        #         x = json.load(f)
        #         for key in x:
        #             if key not in self.config:
        #                 self.config[key] = x[key]

    # def _init_device(self):
    #     use_gpu = self.config.get('gpu', True)
    #     gpu_id = self.config.get('gpu_id', 0)
    #     if use_gpu:
    #         torch.cuda.set_device(gpu_id)
    #     self.config['device'] = torch.device(
    #         "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def convert_all_config_to_strs(self):
        strs = str()
        for idx, (key, value) in enumerate(self.config.items()):
            strs = strs + str(f"{key}: {value}; ")
            if idx != 0 and idx % 4 == 0:
                strs = strs + '\n'
        return strs


    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return self.config.__iter__()


other_arguments = {
    "seed": {
        "type": "int",
        "default": None,
        "help": "random seed"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "num_workers": {
        "type": "int",
        "default": None,
        "help": "num_workers for dataloader"
    },
    "patch_size": {
        "type": "int",
        "default": None,
        "help": "patch_size for sequence transformer"
    },
    "seg_mask_rate":{
        "type": "float",
        "default": None,
        "help": "mask rate for segment sequence masking"
    },
    "collate_func":{
        "type": "str",
        "default": None,
        "help": "choose the collate function"
    },
    "backbone": {
        "type": "str",
        "default": None,
        "help": "specify the pretrained model as backbone"
    },
    "pretrained_path": {
        "type": "str",
        "default": None,
        "help": "specify the pretrained representation model's path, such as 'repository\saved_model\.。。\example.pkL'"
    },
    "wv_dir": {
        "type": "str",
        "default": None,
        "help": "specify the word embedding's path, such as 'repository\saved_model\.。。\example.pkL'"
    },
    "pooler_type": {
        "type": "str",
        "default": None,
        "help": "specify the pooler type for the SimCSE model (cls, first_node, ...)"
    },
    "test_model_load_dict": {
        "type": "str",
        "default": None,
        "help": "specify the evaluated model"
    },
    "freeze": {
        "type": "bool",
        "default": None,
        "help": "choose if freezing the pretrained model"
    },
    "mlp_only_train": {
        "type": "bool",
        "default": None,
        "help": "choose if proccess mlp in the finetune model"
    }
}
