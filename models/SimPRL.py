import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertConfig, BertEmbeddings, BertEncoder, BertPooler
from typing import Optional
import numpy as np
import math
import torch.nn.functional as F


class SimPRL(nn.Module):
    def __init__(self, config):
        super(SimPRL, self).__init__()
        self.config = config
        self.sent_emb = bool(config['sent_emb'])
        self.vocab_size = config['token_num']
        self.bert_config = BertConfig(vocab_size = self.vocab_size, hidden_size = config['hidden_size'], max_position_embeddings = config['max_seq_len'],
                                      num_hidden_layers = config['num_hidden_layers'], num_attention_heads = config['num_attention_heads'],
                                      intermediate_size = config['intermediate_size'], pad_token_id = config['pad_token'])

        self.bert = BERT_4_ID_GPS(config,self.bert_config)

        self.time_encoder = TimeTupleEncoder(time_dim=config['time_dim'])

        self.mlp = MLPLayer(config['out_dim'], config['out_dim'])

        self.dense = nn.Linear(config['hidden_size'] + 3 * config['time_dim'], config['out_dim'])

        self.mlp_only_train = config['mlp_only_train']

        self.loss_fct = nn.CrossEntropyLoss()

        self.pooler_type = config['pooler_type']
        self.mlm_weight = config['mlm_weight']
        self.hard_negative_weight = config['hard_negative_weight']

        self.pooler = Pooler(self.pooler_type, self.bert_config)

        if bool(config['do_mlm']):
            self.lm_head = BertLMPredictionHead(self.bert_config)
        self.sim = Similarity(temp=config['temp'])

        if 'add_psudolabel' in config['identify']:
            self.temporal_prediction = nn.Linear(config['out_dim'], 3)

        config['base_model_params'] = sum(p.numel() for p in self.parameters())

        self.encoder = nn.ModuleList([self.bert,
                                      self.time_encoder
                                      ])

    def forward(self, features, config=None, data_feature=None):
        if self.sent_emb:
            return sentemb_forward(self, self.encoder,
                                               input_ids=features['input_ids'],
                                               input_gps=features['lon_lat'],
                                               attention_mask_seg=features['attention_mask_seg'],
                                               attention_mask_gps=features['attention_mask_gps'],
                                               timestamps=features['departure_timestamp']
                                               )
        else:
            return cl_forward(self, self.encoder,
                input_ids=features['input_ids'],
                input_gps=features['lon_lat'],
                attention_mask_seg=features['attention_mask_seg'],
                attention_mask_gps=features['attention_mask_gps'],
                timestamps = features['departure_timestamp'],
                mlm_input_ids=features['mlm_input_ids'],
                mlm_labels=features['mlm_labels']
            )

def cl_forward(self,
    encoder,
    input_gps = None,
    input_ids=None,
    attention_mask_seg=None,
    attention_mask_gps=None,
    timestamps=None,

    mlm_input_ids=None,
    mlm_labels=None,
):
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    batch_size, num_sent = input_ids.size()[0:2]

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    input_gps = input_gps.view((batch_size * num_sent, -1, input_gps.size(-1))) # (bs * num_sent, len)
    attention_mask_seg = attention_mask_seg.view((-1, attention_mask_seg.size(-1))) # (bs * num_sent len)
    attention_mask_gps = attention_mask_gps.view((-1, attention_mask_gps.size(-1))) # (bs * num_sent len)

    # Get raw sequence output embeddings
    seg_encoder_outputs, gps_encoder_outputs = encoder[0](
        input_gps=input_gps,
        input_ids=input_ids,
        attention_mask_seg=attention_mask_seg,
        attention_mask_gps=attention_mask_gps,
        return_dict=True,
    )

    # Pooling output
    seg_pooler_output = self.pooler(attention_mask_seg, seg_encoder_outputs)
    gps_pooler_output = self.pooler(attention_mask_gps, gps_encoder_outputs)

    seg_pooler_output = seg_pooler_output.view((batch_size, num_sent, seg_pooler_output.size(-1)))  # (bs, num_sent, hidden)
    gps_pooler_output = gps_pooler_output.view((batch_size, num_sent, gps_pooler_output.size(-1)))  # (bs, num_sent, hidden)

    seg_pooler_output = self.dense(torch.cat([seg_pooler_output, encoder[1](timestamps)],dim=-1))
    gps_pooler_output = self.dense(torch.cat([gps_pooler_output, encoder[1](timestamps)],dim=-1))

    # if "cls" in self.pooler_type or self.pooler_type == 'first_node':
    seg_pooler_output = self.mlp(seg_pooler_output)
    gps_pooler_output = self.mlp(gps_pooler_output)

    # Separate representation
    z1, z2 = seg_pooler_output[:,0], seg_pooler_output[:,1]
    z4, z5 = gps_pooler_output[:,0], gps_pooler_output[:,1]

    cos_sim_seg_seg = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    cos_sim_gps_gps = self.sim(z4.unsqueeze(1), z5.unsqueeze(0))
    cos_sim_seg_gps = self.sim(z1.unsqueeze(1), z5.unsqueeze(0))
    cos_sim_gps_seg = self.sim(z4.unsqueeze(1), z2.unsqueeze(0))

    # Calculate loss with hard negatives
    if num_sent == 3:
        # print(1)
        z3 = seg_pooler_output[:, 2]
        z6 = gps_pooler_output[:, 2]

        z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        z1_z6_cos = self.sim(z1.unsqueeze(1), z6.unsqueeze(0))
        z4_z6_cos = self.sim(z4.unsqueeze(1), z6.unsqueeze(0))
        z4_z3_cos = self.sim(z4.unsqueeze(1), z3.unsqueeze(0))
        cos_sim_seg_seg = torch.cat([cos_sim_seg_seg, z1_z3_cos], 1)
        cos_sim_gps_gps = torch.cat([cos_sim_gps_gps, z4_z6_cos], 1)
        cos_sim_seg_gps = torch.cat([cos_sim_seg_gps, z1_z6_cos], 1)
        cos_sim_gps_seg = torch.cat([cos_sim_gps_seg, z4_z3_cos], 1)

        # Note that weights are actually logits of weights
        z3_weight = self.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim_seg_seg.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(input_ids.device)
        cos_sim_seg_seg = cos_sim_seg_seg + weights
        cos_sim_gps_gps = cos_sim_gps_gps + weights
        cos_sim_seg_gps = cos_sim_seg_gps + weights
        cos_sim_gps_seg = cos_sim_gps_seg + weights

    labels = torch.arange(cos_sim_seg_seg.size(0)).long().to(input_ids.device)
    seg_loss = self.loss_fct(cos_sim_seg_seg, labels)
    gps_loss = self.loss_fct(cos_sim_gps_gps, labels)
    seg_gps_loss = self.loss_fct(cos_sim_seg_gps, labels)
    gps_seg_loss = self.loss_fct(cos_sim_gps_seg, labels)
    loss = seg_loss + gps_loss + seg_gps_loss + gps_seg_loss

    # Calculate loss for MLM
    if  mlm_input_ids is not None:
        # print(11)
        mlm_outputs = encoder[0](
            input_ids=mlm_input_ids,
            attention_mask_seg=attention_mask_seg[::num_sent, :],
            return_dict=True,
        )[0] # sequence_output
        prediction_scores = self.lm_head(mlm_outputs)
        masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        loss = loss + self.mlm_weight * masked_lm_loss

    return loss, (z1, z2,z4, z5), (seg_loss.item(), gps_loss.item(), seg_gps_loss.item(), gps_seg_loss.item())


def sentemb_forward(
    self,
    encoder,
    input_ids=None,
    input_gps=None,
    attention_mask_seg=None,
    attention_mask_gps=None,
    timestamps=None,
):
    outputs = encoder[0](
        input_ids=input_ids,
        input_gps=input_gps,
        attention_mask_seg=attention_mask_seg,
        attention_mask_gps = attention_mask_gps,
        return_dict=True
    )
    pooler_output = self.pooler(attention_mask=attention_mask_seg if attention_mask_gps is None else attention_mask_gps, last_hidden=outputs[0])
    if timestamps is not None:
        pooler_output = torch.cat([pooler_output, encoder[1](timestamps)],dim=-1)
        if not self.mlp_only_train:
            pooler_output = self.dense(pooler_output)

    return pooler_output, outputs[0]

class BERT_4_ID_GPS(nn.Module):
    def __init__(self, config, bert_config):
        super().__init__()
        self.num_sent = 1 + int(config['do_mlm']) + int(config['do_hard_neg'])
        self.seg_embedding = BertEmbeddings(bert_config)

        self.in_dim = (config['lat_size'] + config['lon_size'])*2
        # self.gps_embedding = EncTokenEmbedding(self.in_dim, config['hidden_size'])
        self.gps_embedding = GPSEmbedding(self.in_dim, config['hidden_size'])

        self.encoder_4_seg_gps = BertEncoder(bert_config)

    def forward(self,
                input_gps: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask_seg: Optional[torch.Tensor] = None,
                attention_mask_gps: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None
                ):
        sequence_output = []
        if input_ids is not None:
            input_shape = input_ids.size()

            seg_embedding_output = self.seg_embedding(
                input_ids=input_ids
            )

            extended_attention_mask_seg = self.get_extended_attention_mask(attention_mask_seg, input_shape)

            seg_encoder_outputs = self.encoder_4_seg_gps(
                seg_embedding_output,
                attention_mask=extended_attention_mask_seg,
                return_dict=return_dict
            )

            seg_sequence_output = seg_encoder_outputs[0]

            sequence_output.append(seg_sequence_output)
        if input_gps is not None:
            input_shape = input_gps.size()[:-1]
            extended_attention_mask_gps = self.get_extended_attention_mask(attention_mask_gps, input_shape)
            gps_embedding_output = self.gps_embedding(input_gps)
            gps_encoder_outputs = self.encoder_4_seg_gps(
                gps_embedding_output,
                attention_mask=extended_attention_mask_gps,
                return_dict=return_dict
            )
            gps_sequence_output = gps_encoder_outputs[0]
            sequence_output.append(gps_sequence_output)

        return sequence_output

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        """

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(attention_mask.dtype).min
        return extended_attention_mask

    def node_rep(self):
        return F.dropout(self.seg_embedding.word_embeddings.weight, 0.2)


class GPSEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GPSEmbedding, self).__init__()
        self.enc = nn.Linear(in_features=2, out_features=c_in)
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1,
                                   padding_mode='zeros')
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: torch.tensor, [batch_size, seq_len, 2]
        x = self.enc(x)
        x = self.tokenConv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(x)


class TimeTupleEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeTupleEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w1 = nn.Linear(1, time_dim)
        self.w1.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        # self.w1.weight = nn.Parameter((torch.linspace(1, 1 / 10 ** 9, time_dim, dtype=torch.float32)).reshape(time_dim, -1))
        self.w1.bias = nn.Parameter(torch.zeros(time_dim))
        self.w2 = nn.Linear(1, time_dim)
        self.w2.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        # self.w2.weight = nn.Parameter((1 / 10 ** torch.linspace(0, 9, time_dim, dtype=torch.float32)).reshape(time_dim, -1))
        self.w2.bias = nn.Parameter(torch.zeros(time_dim))
        self.w3 = nn.Linear(1, time_dim)
        self.w3.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        # self.w3.weight = nn.Parameter((1 / 10 ** torch.linspace(0, 9, time_dim, dtype=torch.float32)).reshape(time_dim, -1))
        self.w3.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w1.weight.requires_grad = False
            self.w1.bias.requires_grad = False
            self.w2.weight.requires_grad = False
            self.w2.bias.requires_grad = False
            self.w3.weight.requires_grad = False
            self.w3.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        # timestamps: Tensor, shape (batch_size, 3)

        # outputx: Tensor, shape (batch_size, time_dim)
        output1 = torch.cos(self.w1(timestamps[...,0:1]))
        output2 = torch.cos(self.w2(timestamps[...,1:2]))
        output3 = torch.cos(self.w3(timestamps[...,2:3]))
        return torch.cat((output1, output2, output3), dim=-1)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.transform(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls_after_pooler': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    """
    def __init__(self, pooler_type, bert_config, prefix_dim = 0):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_after_pooler", "avg", "raw_last","first_node", 'sum'], "unrecognized pooling type %s" % self.pooler_type
        self.bert_pooler = BertPooler(bert_config)
        self.prefix_dim = prefix_dim

    def dense_embeds(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            return torch.sum(seq * msk, 1) / torch.sum(msk, 1)

    def forward(self, attention_mask, last_hidden):
        # last_hidden = outputs
        # pooler_output = outputs.pooler_output
        # hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type in ['cls_after_pooler']:
            return self.bert_pooler(last_hidden)
        elif self.pooler_type == "first_node":
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "sum":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1)
        elif self.pooler_type == "raw_last":
            pooled_result = (last_hidden * attention_mask.unsqueeze(-1))
            return pooled_result
        else:
            raise NotImplementedError


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        # timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class EncTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(EncTokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1,
                                   padding_mode='zeros')
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: torch.tensor, [batch_size, seq_len, lat_size+lon_size]
        x = self.tokenConv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(x)


