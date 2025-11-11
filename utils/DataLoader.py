import os
import pickle
import pickle as pkl
# import pickle5
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, BatchSampler
from .utils import timestamp2timetuple
from importlib import import_module
from tqdm import tqdm
import time


class Duration_StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class PreTrainDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.adj_gps = data[:,2]
        self.duration = data[:,3]
        self.departure_timestamp = data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.travel_time = data[:,5]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.adj_gps_lens = np.asarray([len(x) for x in self.adj_gps])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.adj_gps[idx],
            self.departure_tuple[idx],
            self.adj_seg_lens[idx],
            self.adj_gps_lens[idx],
            self.departure_timestamp[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)


class TrajectoryProcessingDataset(Dataset):
    def __init__(self, df_folder: str, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode):
        self.data = np.load(df_folder, allow_pickle=True)
        # self.traj_idxs = data[:,0]
        self.adj_seg = self.data[:,1]
        # self.duration = data[:,3]
        self.departure_timestamp = self.data[:,4]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.adj_seg))])
        self.add_cls = add_cls
        self.vocab = vocab
        cache_path = cache_path + f"{mode}.pkl"
        if os.path.exists(cache_path):
            self.traj_list = pickle.load(open(cache_path, 'rb'))
            print('Load dataset from {}'.format(cache_path))
        else:
            self.traj_list = self.data_processing([self.adj_seg, self.departure_timestamp, self.adj_seg_lens], cache_path = cache_path)

        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None

    def __getitem__(self, idx):
        traj_ind = self.traj_list[idx]  # (seq_length, feat_dim)
        mask = self.noise_mask(traj_ind, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                          self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return torch.LongTensor(traj_ind), torch.LongTensor(mask)

    def noise_mask(self, X, masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
        if exclude_feats is not None:
            exclude_feats = set(exclude_feats)

        if distribution == 'geometric':  # stateful (Markov chain)
            if mode == 'separate':  # each variable (feature) is independent
                mask = np.ones(X.shape, dtype=bool)
                for m in range(X.shape[1]):  # feature dimension
                    if exclude_feats is None or m not in exclude_feats:
                        mask[:, m] = self.geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = np.tile(np.expand_dims(self.geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
        elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
            if mode == 'separate':
                mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                        p=(1 - masking_ratio, masking_ratio))
            else:
                mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                                p=(1 - masking_ratio, masking_ratio)), X.shape[1])
        else:
            mask = np.ones(X.shape, dtype=bool)
        if add_cls:
            mask[0] = True  # CLS at 0, set mask=1
        mask[1] = False # avoid nan when calculating CELoss
        return mask

    def geom_noise_mask_single(self, L, lm, masking_ratio):
        keep_mask = np.ones(L, dtype=bool)
        p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * masking_ratio / (
                    1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state

        return keep_mask

    def __len__(self):
        return len(self.traj_list)

    def data_processing(self, origin_data, desc=None, cache_path=None, tmat_path=None):
        print('Processing dataset in TrajectoryProcessingDataset!')
        traj_list = []
        for i in tqdm(range(len(origin_data[0])), desc=desc):
            loc_list = origin_data[0][i]
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            tim_list = origin_data[1][i]
            loc_len = origin_data[2][i]
            tim_list = [tim_list] * loc_len
            new_tim_list = [time.localtime(tim) for tim in tim_list]
            minutes = [new_tim.tm_hour * 60 + new_tim.tm_min + 1 for new_tim in new_tim_list]
            weeks = [new_tim.tm_wday + 1 for new_tim in new_tim_list]
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                tim_list = [tim_list[0]] + tim_list
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks]).transpose((1, 0))
            traj_list.append(traj_fea)
        pickle.dump(traj_list, open(cache_path, 'wb'))
        return traj_list# , temporal_mat_list


class RegFinetuneDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.adj_gps = data[:,2]
        self.duration = data[:,3]
        self.departure_timestamp = data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.travel_time = data[:,5]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.adj_gps_lens = np.asarray([len(x) for x in self.adj_gps])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.departure_tuple[idx],
            self.adj_seg_lens[idx],
            self.travel_time[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)

class CLSFinetuneDataset(Dataset):
    def __init__(self, df_folder: str, data_feature):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.edge_data = data_feature['edgeinfo']
        # self.highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4,
        #                 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8,
        #                 "primary":9, "residential":10, "primary_link":11, "unclassified":12,
        #                 "tertiary":13, "secondary_link":14}
        self.highway = {#'out_of_classes': 0,
                    'living_street': 0, 'motorway_link': 1,
                   'trunk': 2, "secondary": 3, "trunk_link": 4, "tertiary_link": 5,
                   "primary": 6, "residential": 7, "primary_link": 8,
                   "tertiary": 9, "secondary_link": 10}
        self.edge_label = self.get_cls_labesls(self.adj_seg, self.edge_data, self.highway)
        self.departure_timestamp = data[:, 4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.adj_seg_lens[idx],
            self.departure_tuple[idx],
            self.edge_label[idx]
            ]

    def get_cls_labesls(self, adj_seg, edge_data, highway):
        edge_labels = []
        for _, segs in enumerate(tqdm(adj_seg, desc='proccess the segment label ')):
            infos = []
            for x in segs:
                # idx= -100 padding idx,  ['motorway', 'motorway_link'] or 'motorway'
                info = edge_data[x][0]
                if info.startswith("['") and info.endswith("']"):
                    info = eval(info)[0]
                infos.append(highway[info] if info in highway.keys() else -100)
            edge_labels.append(infos)
        return edge_labels

    def __len__(self):
        return len(self.traj_idxs)

class PRFinetuneDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.sim = data[:,2]
        self.wt_sim = data[:,3]

        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.traj_idxs[idx],
            self.adj_seg[idx],
            self.adj_seg_lens[idx],
            self.wt_sim[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)

class RegFinetuneTrajectoryProcessingDataset(TrajectoryProcessingDataset):
    def __init__(self, df_folder: str, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode):
        super().__init__(df_folder, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode)
        self.departure_timestamp = self.data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.travel_time = self.data[:,5]

    def __getitem__(self, idx):
        return torch.LongTensor(self.traj_list[idx]), np.asarray(self.departure_tuple[idx]+1), torch.FloatTensor([self.travel_time[idx]])

class ClsFinetuneTrajectoryProcessingDataset(TrajectoryProcessingDataset):
    def __init__(self, df_folder: str, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode, data_feature):
        super().__init__(df_folder, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode)
        self.departure_timestamp = self.data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.edge_data = data_feature['edgeinfo']
        self.highway = {  # 'out_of_classes': 0,
            'living_street': 0, 'motorway_link': 1,
            'trunk': 2, "secondary": 3, "trunk_link": 4, "tertiary_link": 5,
            "primary": 6, "residential": 7, "primary_link": 8,
            "tertiary": 9, "secondary_link": 10}
        self.edge_label = self.get_cls_labesls(self.adj_seg, self.edge_data, self.highway, add_cls)

    def __getitem__(self, idx):
        return torch.LongTensor(self.traj_list[idx]), np.asarray(self.departure_tuple[idx]+1), torch.FloatTensor([self.edge_label[idx]])

    def get_cls_labesls(self, adj_seg, edge_data, highway, add_cls):
        edge_labels = []
        for _, segs in enumerate(tqdm(adj_seg, desc='proccess the segment label ')):
            infos = []
            if add_cls:
                infos.append(-100)
            for x in segs:
                # idx= -100 padding idx,  ['motorway', 'motorway_link'] or 'motorway'
                info = edge_data[x][0]
                if info.startswith("['") and info.endswith("']"):
                    info = eval(info)[0]
                infos.append(highway[info] if info in highway.keys() else -100)
            edge_labels.append(infos)
        return edge_labels
class SEGTTETaskDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.adj_gps = data[:,2]
        self.duration = data[:,3]
        self.departure_timestamp = data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.travel_time = data[:,5]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.adj_gps_lens = np.asarray([len(x) for x in self.adj_gps])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.departure_tuple[idx],
            self.adj_seg_lens[idx],
            self.travel_time[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)

class GPSDataset(Dataset):
    def __init__(self,  wgs_seq: np.ndarray, merc_seq: np.ndarray, datasets_simi: np.ndarray, max_distance: np.ndarray):
        self.traj_idxs = np.asarray([x for x in range(len(merc_seq))])
        self.traj_len = np.asarray([len(x) for x in merc_seq])
        self.wgs_seq = wgs_seq
        self.merc_seq = merc_seq
        self.max_distance = max_distance
        self.datasets_simi = torch.FloatTensor(np.stack(datasets_simi))
        self.datasets_simi = (self.datasets_simi + self.datasets_simi.T) / self.max_distance

    def __getitem__(self, idx):
        return [self.traj_idxs[idx], self.traj_len[idx], self.merc_seq[idx], self.datasets_simi[idx]]

    def __len__(self):
        return len(self.traj_len)


class SegBatchSampler():
    def __init__(self, count, adj_segments_lens, idxs, batch_size, drop_last = True):
        self.count = count
        self.adj_segments_lens = adj_segments_lens
        self.batch_size = batch_size
        self.idxs = list(idxs)
        self.drop_last = drop_last

        np.random.shuffle(self.idxs)
        self.chunk_size = self.batch_size * 100
        self.chunks = (self.count + self.chunk_size - 1) // self.chunk_size
        # re-arrange indices to minimize the padding
        for i in range(self.chunks):
            partial_indices = self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size]
            partial_indices.sort(key=lambda x: self.adj_segments_lens[x], reverse=True)
            self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size] = partial_indices

        # yield batch, and the last has been dropped
        self.batches = (self.count - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        for i in range(self.batches):
            yield self.idxs[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


class GPSBatchSampler():
    def __init__(self, count, lens, idxs, batch_size, drop_last = True):
        self.count = count
        self.lens = lens
        self.batch_size = batch_size
        self.idxs = list(idxs)
        self.drop_last = drop_last

        # self.chunk_size = self.batch_size * 100
        # self.chunks = (self.count + self.chunk_size - 1) // self.chunk_size
        # # re-arrange indices to minimize the padding
        # for i in range(self.chunks):
        #     partial_indices = self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size]
        #     partial_indices.sort(key=lambda x: self.lens[x], reverse=True)
        #     self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size] = partial_indices

        # yield batch, and the last has been dropped
        self.batches = (self.count - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        for i in range(self.batches):
            yield self.idxs[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


class GPSBatchSampler_random_shuffle():
    def __init__(self, count, lens, idxs, batch_size, drop_last = True):
        self.count = count
        self.lens = lens
        self.batch_size = batch_size
        self.idxs = list(idxs)
        self.drop_last = drop_last

        np.random.shuffle(self.idxs)
        self.chunk_size = self.batch_size * 100
        self.chunks = (self.count + self.chunk_size - 1) // self.chunk_size
        # re-arrange indices to minimize the padding
        for i in range(self.chunks):
            partial_indices = self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size]
            partial_indices.sort(key=lambda x: self.lens[x], reverse=True)
            self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size] = partial_indices

        # yield batch, and the last has been dropped
        self.batches = (self.count - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        for i in range(self.batches):
            yield self.idxs[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_feature_data(config):
    data_feature ={}
    # if 'MLP_SIM' == config['model']:
    #     with open(os.path.join(config['base_dir'], config['data_dir'], f"traj_simi_dict_{config['simi_fn']}.pkl"),
    #               'rb') as fh:
    #         data_feature['train'], data_feature['val'], data_feature['test'], data_feature[
    #             'max_distance'] = pkl.load(fh)
    #     return data_feature

    if 'TTE' in config['model']:
        # travel time is normalized with scaler
        data_feature['time_standard'] = Duration_StandardScaler(mean=config['time_mean'], std=config['time_std'])

        data_feature['lenth_standard'] = StandardScaler()
        data_feature['lenth_standard'].fit([[0, 0]])
        data_feature['lenth_standard'].mean_ = config['length_sumlength_mean']
        data_feature['lenth_standard'].scale_ = config['length_sumlength_std']

        data_feature['gps_standard'] = StandardScaler()
        data_feature['gps_standard'].fit([[0, 0, 0, 0]])
        data_feature['gps_standard'].mean_ = config['two_ends_gps_mean']
        data_feature['gps_standard'].scale_ = config['two_ends_gps_std']

    if 'MLP_PR' in config['model']:
        # travel time is normalized with scaler
        data_feature['sim_standard'] = Duration_StandardScaler(mean=config['sim_mean'], std=config['sim_std'])
    if "START" in config['model']:
        from models.START import WordVocab
        data_feature['vocab'] = WordVocab.load_vocab(os.path.join(config['base_dir'], config['vocab_path']))

    if 'skipgram' in config['identify'] and '4Embeds' in config['model']:
        print(f"load skipgram of {config['dataset']}")
        config['wv_dir'] = f"repository/saved_model/SkipGram/{config['dataset']}/final_model.pkl"
        data_feature['wv'] = torch.load(os.path.join(config['base_dir'], config['wv_dir'])).wv
    elif 'node2vec' in config['identify'] and '4Embeds' in config['model']:
        print(f"load node2vec of {config['dataset']}")
        config['wv_dir'] = f"processed_data/{config['dataset']}/Node2Vec_{config['dataset']}_edges_embeds.pkl"   # python 3.9才能加载，否则报错TypeError: __randomstate_ctor() takes from 0 to 1 positional arguments but 2 were given
        data_feature['wv'] = torch.load(os.path.join(config['base_dir'], config['wv_dir'])).wv    # python 3.9才能加载，否则报错TypeError: __randomstate_ctor() takes from 0 to 1 positional arguments but 2 were given

    # basic edge and node attributes in a specific city data
    with open(os.path.join(config['base_dir'], config['edges_dir']), 'rb') as f:
        data_feature['edgeinfo'] = pkl.load(f)
    with open(os.path.join(config['base_dir'], config['nodes_dir']), 'rb') as f:
        data_feature['nodeinfo'] = pkl.load(f)

    return data_feature

def get_dataset(config, data_feature):
    return getattr(import_module('utils.DataLoader'), f"get_{config['task']}_{config['model']}_dataloader")(config, data_feature)

def get_end2end_train_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            full_data[phase] = SEGTTETaskDataset(
                df_folder=os.path.join(config['base_dir'], f"processed_data/{config['dataset']}/pretrain/pretrain.npy"))
        else:
            full_data[phase] = SEGTTETaskDataset(df_folder=os.path.join(config['base_dir'], f"processed_data/{config['dataset']}/finetune/finetune_from_pretrain_random/{phase}.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_reg_finetune_MLP_TTE_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = RegFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_reg_finetune_MLP_TTE4Embeds_dataloader(config, data_feature):
    return get_reg_finetune_MLP_TTE_dataloader(config, data_feature)

def get_cls_finetune_MLP_CLS_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = CLSFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"), data_feature=data_feature)
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_cls_finetune_MLP_CLS4Embeds_dataloader(config, data_feature):
    return get_cls_finetune_MLP_CLS_dataloader(config, data_feature)

def get_reg_finetune_TTE4START_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = RegFinetuneTrajectoryProcessingDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"),
                                                       vocab=data_feature['vocab'], add_cls = config['add_cls'], cache_path = config['cache_path'],
                                                       masking_ratio=config['masking_ratio'], masking_mode=config['masking_mode'],
                                                       distribution=config['distribution'], avg_mask_len =config['avg_mask_len'], mode = phase)
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_cls_finetune_CLS4START_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = ClsFinetuneTrajectoryProcessingDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"),
                                                       vocab=data_feature['vocab'], add_cls = config['add_cls'], cache_path = os.path.join(config['base_dir'],config['cache_path']),
                                                       masking_ratio=config['masking_ratio'], masking_mode=config['masking_mode'],
                                                       distribution=config['distribution'], avg_mask_len =config['avg_mask_len'], mode = phase, data_feature = data_feature)
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_rrl_pretrain_SimPRL_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['pretrain', 'pretrain_eval']:
        full_data[phase] = PreTrainDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn
        )
    return dataloader.copy()

def get_rrl_pretrain_JCLRNT_dataloader(config, data_feature):
    return get_rrl_pretrain_SimPRL_dataloader(config, data_feature)

def get_rrl_pretrain_START_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['pretrain', 'pretrain_eval']:
        full_data[phase] = TrajectoryProcessingDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"),
                                                       vocab=data_feature['vocab'], add_cls = config['add_cls'], cache_path = config['cache_path'],
                                                       masking_ratio=config['masking_ratio'], masking_mode=config['masking_mode'],
                                                       distribution=config['distribution'], avg_mask_len =config['avg_mask_len'], mode = phase)
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn
        )
    return dataloader.copy()

def get_pr_finetune_MLP_PR_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = PRFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}_PR_top_4.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_pr_finetune_MLP_SIM_dataloader(config, data_feature):
    full_data = {}
    dataloader = {}

    for phase in ['train', 'val', 'test']:
        data_df = np.load(os.path.join(config['base_dir'], config['data_dir'], f"{phase}_traj_simi_dict_hausdorff.npy"), allow_pickle=True)
        wgs_seq = data_df[0]
        merc_seq = data_df[1]
        sims = data_df[2]
        full_data[phase] = GPSDataset(wgs_seq=wgs_seq, merc_seq = merc_seq, datasets_simi=sims, max_distance=config['max_distance'])
        data_feature[f'{phase}_sims'] = full_data[phase].datasets_simi
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=GPSBatchSampler(count=len(full_data[phase]),lens=full_data[phase].traj_len,idxs=full_data[phase].traj_idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

# def collate_fn_select(data, config, data_feature):
#     return getattr(import_module('utils.model_collate_fn'),config['collate_func'])(data=data, config=config, data_feature=data_feature)


class CollateFnWrapper:
    def __init__(self, config, data_feature):
        self.config = config
        self.data_feature = data_feature

    def __call__(self, batch):
        return getattr(import_module('utils.model_collate_fn'),self.config['collate_func'])(data=batch, config=self.config, data_feature=self.data_feature)