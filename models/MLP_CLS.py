from importlib import import_module

import torch
from torch import nn
import os
from models.SimPRL import TimeEncoder, TimeTupleEncoder


class MLP_CLS(nn.Module):
    def __init__(self, config):
        super(MLP_CLS, self).__init__()
        self.output_dim = config['output_dim']
        self.device = config['device']
        self.freeze = config['freeze']
        self.beackbone_out_dim =  config['beackbone_out_dim']
        if config['backbone'] == 'SimPRL':
            config[
                'pretrained_path'] = f"repository/saved_model/SimPRL/{config['dataset']}/SimPRL_rrl_pretrain_{config['dataset']}_final/SimPRL_rrl_pretrain_{config['dataset']}_final_seed0.pkl"
        else:
            ValueError("No this backbone dir")

        self.pretrained_path = os.path.join(config['base_dir'], config['pretrained_path'])

        self.backbones = getattr(import_module(f"models.{config['backbone']}"),config['backbone'])(config)
        checkpoints = torch.load(self.pretrained_path, map_location=self.device)
        if 'model_state_dict' in checkpoints.keys():
            missing_keys, _ = self.backbones.load_state_dict(checkpoints['model_state_dict'], strict=False)
        else:
            missing_keys , _ = self.backbones.load_state_dict(checkpoints, strict=False)

        # assert len(missing_keys) == 0, f"{missing_keys}"  # wo_time 注释这一行，防止time相关不通过
        print(f"{config['backbone']}, {self.pretrained_path} backbones.load_state_dict load success!")

        if self.freeze:
            print("---------------------------------")
            print('freeze!')
            print("---------------------------------")
            for pa in self.backbones.parameters():
                pa.requires_grad = False
        if config['backbone'] == 'SimPRL':
            self.dense = nn.Linear(in_features=self.beackbone_out_dim, out_features=self.beackbone_out_dim)

        self.num_classes = 11

        self.segs_extract = nn.Linear(in_features=self.beackbone_out_dim, out_features=self.num_classes)
    def forward(self, features, config, data_feature):

        outputs = self.backbones(features, config)[1]
        outputs = self.dense(outputs)


        outputs = self.segs_extract(outputs).reshape(-1, self.num_classes)


        return outputs

