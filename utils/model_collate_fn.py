import os
import time
import torch
import numpy as np
from .utils import SEGRandomMask, GPSShift, TimeSlide


highway = {'living_street': 1, 'motorway': 2, 'motorway_link': 3, 'plannned': 4, 'trunk': 5, "secondary": 6,
           "trunk_link": 7, "tertiary_link": 8, "primary": 9, "residential": 10, "primary_link": 11, "unclassified": 12,
           "tertiary": 13, "secondary_link": 14}
node_type = {'turning_circle': 1, 'traffic_signals': 2, 'crossing': 3, 'motorway_junction': 4, "mini_roundabout": 5}


def SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    num_edge, cls_token, pad_token, mask_token, seg_pad_rate = config['edges'], config['cls_token'], config[
        'pad_token'], config['mask_token'], config['seg_pad_rate']
    # lon_size, lat_size, round_num = config['lon_size'], config['lat_size'], config['round_num']
    mlm_probability = config['mlm_probability']
    mlm_input_ids, mlm_labels = None, None
    seg_id = []
    pos_seg_id = []
    gps_id = []
    pos_gps_id = []
    departure_timestamp = []
    pos_departure_timestamp = []
    seg_len = []
    gps_len = []
    for ind, l in enumerate(data):
        seg_id.append(l[1])
        pos_seg_id.append(SEGRandomMask(l[1], l[4], seg_pad_rate, pad_token))
        gps_id.append(l[2])
        pos_gps_id.append(GPSShift(l[2]))
        departure_timestamp.append(l[3])
        pos_departure_timestamp.append(TimeSlide(l[-1]))
        seg_len.append(l[4])
        gps_len.append(l[5])

    seg_len = np.asarray(seg_len)
    gps_len = np.asarray(gps_len)
    # departure_timestamp = torch.unsqueeze(torch.FloatTensor(departure_timestamp),dim=1)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    pos_departure_timestamp = torch.FloatTensor(np.asarray(pos_departure_timestamp))
    max_seq_length = seg_len.max()
    max_gps_length = gps_len.max()

    # pad the original and positive items and mask attention for seg and gps
    # construct the matrix with [batch_size x max_seq_len]
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the segid with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_token)
    padded_seg_id[mask] = np.concatenate(seg_id)
    padded_seg_id = torch.LongTensor(padded_seg_id)

    padded_pos_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_token)
    padded_pos_seg_id[mask] = np.concatenate(pos_seg_id)
    padded_pos_seg_id = torch.LongTensor(padded_pos_seg_id)

    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1.)

    mask = np.arange(max_gps_length) < gps_len[:, None]
    padded_gps_id = np.zeros((*mask.shape, 2), dtype=np.float32)
    padded_gps_id[mask] = np.concatenate(gps_id)
    padded_gps_id = torch.FloatTensor(padded_gps_id)

    padded_pos_gps_id = np.zeros((*mask.shape, 2), dtype=np.float32)
    padded_pos_gps_id[mask] = np.concatenate(pos_gps_id)
    padded_pos_gps_id = torch.FloatTensor(padded_pos_gps_id)

    mask_encoder_gps = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_gps.masked_fill_(torch.BoolTensor(mask), value=1.)

    padded_pos_seg_id = torch.unsqueeze(padded_pos_seg_id, dim=1)
    padded_seg_id = torch.unsqueeze(padded_seg_id, dim=1)
    padded_pos_gps_id = torch.unsqueeze(padded_pos_gps_id, dim=1)
    padded_gps_id = torch.unsqueeze(padded_gps_id, dim=1)
    departure_timestamp = torch.unsqueeze(departure_timestamp, dim=1)
    pos_departure_timestamp = torch.unsqueeze(pos_departure_timestamp, dim=1)

    # constrict the original, positive samples as [batch, 2, dim, ...]
    mask_encoder_seg = torch.unsqueeze(mask_encoder_seg, dim=1).repeat(1, 2, 1)
    mask_encoder_gps = torch.unsqueeze(mask_encoder_gps, dim=1).repeat(1, 2, 1)
    # departure_timestamp = torch.unsqueeze(departure_timestamp, dim=1).repeat(1, 2, 1)
    departure_timestamp = torch.cat([departure_timestamp, pos_departure_timestamp], dim=1)
    padded_seg_id = torch.cat([padded_seg_id, padded_pos_seg_id], dim=1)
    padded_gps_id = torch.cat([padded_gps_id, padded_pos_gps_id], dim=1)

    if config['do_hard_neg'] and not config['sent_emb']:
        # construct the neg matrix of each part as [natch_size, 1, dim, ...]
        # padded_neg_seg_id = torch.cat([padded_seg_id[1:, 0:1, ...], padded_seg_id[0, 0:1, None]], dim=0)
        # padded_neg_gps_id = torch.cat([padded_gps_id[1:, 0:1, ...], padded_gps_id[0, 0:1, None]], dim=0)
        padded_neg_seg_id = torch.cat([padded_seg_id[1:, 1:2, ...], padded_seg_id[0, 1:2, None]], dim=0)
        padded_neg_gps_id = torch.cat([padded_gps_id[1:, 1:2, ...], padded_gps_id[0, 1:2, None]], dim=0)
        # to avoid the loss calculation between anchor and itself as false negatives, take the anchor and the augmented positive sample, they are calculated as hard negatives
        negs_mask_encoder_seg = torch.cat([mask_encoder_seg[1:, 0:1, ...], mask_encoder_seg[0, 0:1, None]], dim=0)
        negs_mask_encoder_gps = torch.cat([mask_encoder_gps[1:, 0:1, ...], mask_encoder_gps[0, 0:1, None]], dim=0)
        neg_dateinfo = torch.cat([departure_timestamp[1:, 0:1, ...], departure_timestamp[0, 0:1, None]], dim=0)

        # constrict the original, positive, negtive samples as [batch, 3, dim, ...]
        padded_seg_id = torch.cat([padded_seg_id, padded_neg_seg_id], dim=1)
        padded_gps_id = torch.cat([padded_gps_id, padded_neg_gps_id], dim=1)

        mask_encoder_seg = torch.cat([mask_encoder_seg, negs_mask_encoder_seg], dim=1)
        mask_encoder_gps = torch.cat([mask_encoder_gps, negs_mask_encoder_gps], dim=1)
        departure_timestamp = torch.cat([departure_timestamp, neg_dateinfo], dim=1)

    if config['do_mlm'] and not config['sent_emb']:
        def mask_tokens(inputs: torch.Tensor):
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            pad_len = max_seq_length - len(inputs)

            masked_indices = torch.rand(labels.shape) <= mlm_probability

            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = torch.LongTensor([mask_token])

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(num_edge, labels.shape, dtype=torch.int)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return [torch.cat([inputs, torch.full((pad_len,), dtype=torch.int, fill_value=pad_token)]),
                    torch.cat([labels, torch.full((pad_len,), fill_value=-100, dtype=torch.int)])]

        mlm_input_ids = []
        mlm_labels = []
        for y, z in [mask_tokens(torch.IntTensor(x)) for x in seg_id]:
            mlm_input_ids.append(y)
            mlm_labels.append(z)

        mlm_input_ids = torch.stack(mlm_input_ids)
        mlm_labels = torch.stack(mlm_labels)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((*padded_seg_id.shape[0:2], 1), fill_value=cls_token), padded_seg_id],
                                  dim=-1)
        mask_encoder_seg = torch.cat([torch.ones((*mask_encoder_seg.shape[0:2], 1)), mask_encoder_seg], dim=-1)

        if config['do_mlm'] and not config['sent_emb']:
            # add the first token of the seg_id as the mask token, to avoid the nan in ce loss calculation in some cases when mlm_imnput_ids has no mask_token
            mlm_input_ids = torch.cat([torch.full((batch_size, 1), fill_value=mask_token), mlm_input_ids], dim=-1)
            mlm_labels = torch.cat([padded_seg_id[:, 0, 0:1], mlm_labels], dim=-1)
            # mlm_input_ids = torch.cat([torch.full((batch_size,1),fill_value=cls_token), mlm_input_ids],dim=-1)
            # mlm_labels = torch.cat([torch.full((batch_size,1),fill_value=-100), mlm_labels],dim=-1)

    if config['sent_emb']:
        time = [d[-1] for d in data]
        return {'input_ids': padded_seg_id,
                'attention_mask_seg': mask_encoder_seg, 'lon_lat': None, 'attention_mask_gps': None,
                'mlm_input_ids': None, 'mlm_labels': None,
                'departure_timestamp': departure_timestamp}, torch.FloatTensor(time)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg,
            "departure_timestamp": departure_timestamp,
            "lon_lat": padded_gps_id, 'attention_mask_gps': mask_encoder_gps,
            'mlm_input_ids': mlm_input_ids, 'mlm_labels': mlm_labels}, torch.Tensor([])

def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    # max_len = max_len# or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def TTE4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    seg_id = []
    departure_timestamp = []
    seg_len = []
    time = []
    for ind, l in enumerate(data):
        seg_id.append(l[1])
        # departure_timestamp.append(timestamp2timetuple(l[2]))
        departure_timestamp.append(l[2])
        seg_len.append(l[3])
        time.append(l[4])

    seg_len = np.asarray(seg_len)
    # departure_timestamp = torch.unsqueeze(torch.FloatTensor(departure_timestamp),dim=1)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max()

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(seg_id)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None, 'departure_timestamp': departure_timestamp}, torch.FloatTensor(
        time)


def REG4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    seg_id = []
    departure_timestamp = []
    seg_len = []
    spd = []
    for ind, l in enumerate(data):
        seg_id.append(l[1])
        departure_timestamp.append(l[2])
        seg_len.append(l[3])
        spd.append(l[-1])

    seg_len = np.asarray(seg_len)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max()  # if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(seg_id)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None, 'departure_timestamp': departure_timestamp}, torch.FloatTensor(
        spd)


def CLS4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    linkids = []
    link_labels = []
    inds = []
    lens = []
    departure_timestamp = []
    for ind, l in enumerate(data):
        linkids.append(np.asarray(l[1]))
        link_labels.append(np.asarray(l[-1]))
        inds.append(l[0])
        lens.append(l[2])
        departure_timestamp.append(l[3])

    seg_len = np.asarray(lens)
    # departure_timestamp = torch.unsqueeze(torch.FloatTensor(departure_timestamp),dim=1)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max()  # if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(linkids)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    padded_linkids_labels = np.full(mask.shape, fill_value=-100, dtype=np.int16)
    padded_linkids_labels[mask] = np.concatenate(link_labels)
    padded_linkids_labels = torch.LongTensor(padded_linkids_labels)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)
        padded_linkids_labels = torch.cat([torch.full((batch_size, 1), fill_value=-100), padded_linkids_labels], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None,
            'departure_timestamp': departure_timestamp}, padded_linkids_labels.reshape(-1)

