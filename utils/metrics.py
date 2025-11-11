import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
from scipy.stats import kendalltau, spearmanr
import pandas as pd
import torch.nn.functional as F


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        mse = np.nan_to_num(mse)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        mae = np.nan_to_num(mae)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
        mape = np.nan_to_num(mask * mape)
        mape = np.nan_to_num(mape)
        return np.mean(mape)


def calculate_regression_metrics(preds: torch.Tensor, labels: torch.Tensor):
    # preds and labels must be torch.tensor
    try:
        preds = preds.cpu().detach().reshape(-1)
        labels = labels.reshape(-1)
        mape = torch.mean(torch.abs(torch.divide(torch.sub(preds, labels), labels + 1e-5)))
        mse = torch.mean(torch.square(torch.sub(preds, labels)))
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(torch.sub(preds, labels)))
        pearsonrs = pearsonr(preds, labels)

    except Exception as e:
        print(e)
        mae = 0
        mape = 0
        rmse = 0
        pearsonrs = (None, None)

    return {'MAE': float(mae), 'MAPE': float(mape), 'RMSE': float(rmse), 'PEARR':pearsonrs[0], 'PEARP': pearsonrs[1]}

def calculate_classification_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, ) or (num_samples, classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    labels = labels.reshape(-1)
    labels_index = torch.where(labels != -100)[0]
    preds = preds.reshape(-1)

    if len(preds.shape) == 2:
        preds = torch.argmax(preds, dim=-1)
    preds = preds.cpu().detach()
    preds = preds.reshape(-1)[labels_index]
    labels = labels[labels_index]

    accuracy_scores = accuracy_score(y_true=labels, y_pred=preds)
    precision_scores = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall_scores = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1_scores = f1_score(y_true=labels, y_pred=preds, average='macro')

    return {'accuracy_scores': accuracy_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'f1_scores': f1_scores}

def calculate_multi_classification_metrics(preds: list, labels: list):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}
    for idx in range(len(preds)):
        label = labels[idx]
        pred = preds[idx]

        label = label.reshape(-1)
        label_index = torch.where(label != -100)[0]
        if len(pred.shape) == 2:
            pred = torch.argmax(pred, dim=-1)
        pred = pred.cpu().detach().reshape(-1)
        pred = pred[label_index]
        label = label[label_index]
        # print(f'\n preds: {pred[:5]}')
        # print(f'labels: {label[:5]}')

        accuracy_scores = accuracy_score(y_true=label, y_pred=pred)
        precision_scores = precision_score(y_true=label, y_pred=pred, average='macro')
        recall_scores = recall_score(y_true=label, y_pred=pred, average='macro')
        f1_scores = f1_score(y_true=label, y_pred=pred, average='macro')
        metrics[f'accuracy_scores_{idx}'] = accuracy_scores
        metrics[f'precision_scores_{idx}'] = precision_scores
        metrics[f'recall_scores_{idx}'] = recall_scores
        metrics[f'f1_scores_{idx}'] = f1_scores

    return metrics

def calculate_regression_classification_metrics(preds: list, labels: list):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}

    label = labels[0]
    pred = preds[0]

    label = label.reshape(-1)
    label_index = torch.where(label != -100)[0]
    if len(pred.shape) == 2:
        pred = torch.argmax(pred, dim=-1)
    pred = pred.cpu().detach().reshape(-1)
    pred = pred[label_index]
    label = label[label_index]

    accuracy_scores = accuracy_score(y_true=label, y_pred=pred)
    precision_scores = precision_score(y_true=label, y_pred=pred, average='macro')
    recall_scores = recall_score(y_true=label, y_pred=pred, average='macro')
    f1_scores = f1_score(y_true=label, y_pred=pred, average='macro')
    metrics[f'accuracy_scores'] = accuracy_scores
    metrics[f'precision_scores'] = precision_scores
    metrics[f'recall_scores'] = recall_scores
    metrics[f'f1_scores'] = f1_scores

    label = labels[1]
    label = label.reshape(-1)
    label_index = torch.where(label != -100)[0]
    pred = preds[1]
    if len(pred.shape) == 2:
        pred = torch.argmax(pred, dim=-1).float()
    pred = pred.cpu().detach().reshape(-1)
    pred = pred[label_index]
    label = label[label_index]
    # print(f'preds: {pred[:10]}')
    # print(f'labels: {label[:10]}')
    try:
        mape = torch.mean(torch.abs(torch.divide(torch.sub(pred, label), label + 1e-5)))
        mse = torch.mean(torch.square(torch.sub(pred, label)))
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(torch.sub(pred, label)))
        pearsonrs = pearsonr(pred, label)

    except Exception as e:
        print(e)
        mae = 0
        mape = 0
        rmse = 0
        # pearsonrs = (None, None)
        pearsonrs = (0, 0)

    metrics['MAE'] = float(mae)
    metrics['MAPE'] = float(mape)
    metrics['RMSE'] = float(rmse)
    metrics['PEARR'] = pearsonrs[0]
    metrics['PEARP'] = pearsonrs[1]

    return metrics


def calculate_multi_classification_acc_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}

    # label = labels[0]
    # pred = preds[0]

    labels = labels.reshape(-1)
    label_index = torch.where(labels != -100)[0]

    preds = preds.cpu().detach()
    preds = preds[label_index]
    labels = labels[label_index]

    if len(preds.shape) == 2:
        preds = torch.argmax(preds, dim=-1).reshape(-1).float()
    metrics[f'accuracy_scores'] = accuracy_score(y_true=labels, y_pred=preds)
    return metrics

def calculate_multi_classification_sum_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}

    # label = labels[0]
    # pred = preds[0]

    # labels = labels.reshape(-1)
    # label_index = torch.where(labels != -100)[0]
    #
    # preds = preds.cpu().detach()
    # preds = preds[label_index]
    # labels = labels[label_index]

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = F.softmax(preds, dim=-1)
    metrics['auc_ovo'] = roc_auc_score(labels, preds, multi_class='ovo')
    metrics['auc_ovr'] = roc_auc_score(labels, preds, multi_class='ovr')

    if len(preds.shape) == 2:
        preds = torch.argmax(preds, dim=-1).reshape(-1).float()
    metrics[f'f1_scores_macro'] = f1_score(y_true=labels, y_pred=preds, average='macro')
    metrics[f'f1_scores_micro'] = f1_score(y_true=labels, y_pred=preds, average='micro')
    metrics[f'balanced_accuracy'] = balanced_accuracy_score(y_true=labels, y_pred=preds)
    metrics[f'accuracy_scores'] = accuracy_score(y_true=labels, y_pred=preds)
    return metrics

def calculate_path_rank_pre_metrics(preds: torch.Tensor, labels: torch.Tensor):
    # preds and labels must be torch.tensor

    preds = preds.cpu().detach().reshape(-1)
    labels = labels.reshape(-1)

    mae = torch.mean(torch.abs(torch.sub(preds, labels)))
    mare = mae / torch.mean(torch.abs(labels))

    return {'MAE': float(mae), 'MARE': float(mare)}

def calculate_path_rank_sum_metrics(preds: torch.Tensor, labels: torch.Tensor, idxs=None):
    # preds and labels must be torch.tensor
    preds = torch.cat(preds, dim=0).reshape(-1)
    labels = torch.cat(labels, dim=0)
    idxs = torch.cat(idxs, dim=0).cpu()

    mae = torch.mean(torch.abs(torch.sub(preds, labels)))
    mare = mae / torch.mean(torch.abs(labels))

    # 创建 DataFrame
    df = pd.DataFrame(np.asarray(torch.stack([idxs, labels, preds], dim=1)), columns=['idx', 'ground_truth', 'predict'])

    # 按 idx 分组并计算相关系数
    kendall_tau_values = []
    spearman_corr_values = []

    for idx, group in df.groupby('idx'):
        tau, _ = kendalltau(group['ground_truth'], group['predict'])
        rho, _ = spearmanr(group['ground_truth'], group['predict'])

        kendall_tau_values.append(tau)
        spearman_corr_values.append(rho)

    # 计算均值
    mean_kendall_tau = pd.Series(kendall_tau_values).mean()
    mean_spearman_corr = pd.Series(spearman_corr_values).mean()

    return {'MAE': float(mae), 'MARE': float(mare), 'kendall_tau': float(mean_kendall_tau),'spearman_corr': float(mean_spearman_corr)}


def top_k_accuracy(top_k, predictions, labels):
    assert len(predictions) == len(labels)
    total = 0
    correct = 0
    for i in range(len(predictions)):
        total += 1
        prediction = []
        for j, k in enumerate(predictions[i]):
            prediction.append([j, k]) # k is the value
        prediction.sort(key = lambda x: -x[1])
        for j, _ in prediction[:top_k]:
            if j == labels[i]:
                correct += 1
                break
    return correct/total

def calculate_traj_similarity(preds: torch.Tensor, labels: torch.Tensor, idxs):
    metrics = {}
    preds = torch.cat(preds)
    pred_l1_simi = torch.cdist(preds, preds, 1)

    metrics['top_5_5'] = hitting_ratio(pred_l1_simi, labels, 5, 5)
    metrics['top_20_20'] = hitting_ratio(pred_l1_simi, labels, 20, 20)
    metrics['top_20_5'] = hitting_ratio(pred_l1_simi, labels, 20, 5)
    metrics['top_10_10'] = hitting_ratio(pred_l1_simi, labels, 10, 10)
    metrics['top_10_5'] = hitting_ratio(pred_l1_simi, labels, 10, 5)
    return metrics

def calculate_traj_similarity_everybatch(preds: torch.Tensor, labels: torch.Tensor):
    metrics = {}
    pred_l1_simi = torch.cdist(preds, preds, 1)
    truth_l1_simi = labels

    metrics['top_5_5'] = hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
    metrics['top_20_20'] = hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
    metrics['top_20_5'] = hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)
    metrics['top_10_10'] = hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
    metrics['top_10_5'] = hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 5)
    return metrics

def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
    # hitting ratio and recall metrics. see NeuTraj paper
    # the overlap percentage of the topk predicted results and the topk ground truth
    # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

    # preds = [batch_size, class_num], tensor, element indicates the probability
    # truths = [batch_size, class_num], tensor, element indicates the probability
    assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

    _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim=1, largest=False)
    _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim=1, largest=False)

    preds_k_idx = preds_k_idx.cpu()
    truths_k_idx = truths_k_idx.cpu()

    tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])

    return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])

def calculate_loc_acc(values: list):
    pred, targets, targets_mask, num_active_l = values

    mask_label = targets[targets_mask]  # (num_active, )
    lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
    correct_l = mask_label.eq(lm_output).sum().item()

    return {"Loc_acc": (correct_l / num_active_l * 100).detach().cpu().item()}

def transfer_loss(values: list):

    return {"loss": values[0].detach().cpu().item()}

def calculate_align_unif_metrics(values: list):
    # preds: x; labels: x_plus
    # must double float, otherwise -inf after uniform caculation
    # allgns = align_loss(preds, labels).cpu().item()
    # unifms = ((uniform_loss(preds) + uniform_loss(labels)) / 2).cpu().item()
    z1, z2 = values[0],values[1]
    z4, z5 = values[2], values[3]
    #
    # z1, z2 = L2Norm(values[0]),L2Norm(values[1])
    # z4, z5 = L2Norm(values[2]), L2Norm(values[3])
    # return {'seg_allgns': align_loss(preds[0], preds[1]).cpu().item(),
    return {'seg_allgns': align_loss(z1, z2).detach().cpu().item(),
            'seg_unifms': ((uniform_loss(z1) + uniform_loss(z2)) / 2).detach().cpu().item(),
            # 'gps_allgns': align_loss(labels[0], labels[1]).cpu().item(),
            'gps_allgns': align_loss(z4, z5).detach().cpu().item(),
            'gps_unifms': ((uniform_loss(z4) + uniform_loss(z5)) / 2).detach().cpu().item(),
            # 'seg_gps_allgns': align_loss(preds[0], labels[1]).cpu().item(),
            'seg_gps_allgns': align_loss(z1, z5).detach().cpu().item(),
            'seg_gps_unifms': ((uniform_loss(z1) + uniform_loss(z5)) / 2).detach().cpu().item(),
            # 'gps_seg_allgns': align_loss(labels[0], preds[1]).cpu().item(),
            'gps_seg_allgns': align_loss(z4, z2).detach().cpu().item(),
            'gps_seg_unifms': ((uniform_loss(z4) + uniform_loss(z2)) / 2).detach().cpu().item(),
            "z1": z1.cpu().detach(),
            "z2": z2.cpu().detach()
            }

def calculate_wo_gps_align_unif_metrics(values: list):
    # preds: x; labels: x_plus
    # must double float, otherwise -inf after uniform caculation
    # allgns = align_loss(preds, labels).cpu().item()
    # unifms = ((uniform_loss(preds) + uniform_loss(labels)) / 2).cpu().item()
    z1, z2, z3 = values[0],values[1], values[2]

    return {'seg_allgns': align_loss(z1, z2).detach().cpu().item(),
            'seg_unifms': ((uniform_loss(z1) + uniform_loss(z2)) / 2).detach().cpu().item(),
            'seg_allgns_neg': align_loss(z1, z3).detach().cpu().item(),
            'seg_unifms_neg': ((uniform_loss(z1) + uniform_loss(z3)) / 2).detach().cpu().item(),
            }


# These metrics/losses are useful for:
# (as metrics) quantifying encoder feature distribution properties,
# (as losses) directly training the encoder.

# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x.double() - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=1):
    return torch.pdist(x.double(), p=2).pow(2).mul(-t).exp().mean().log()

def L2Norm(x, dim = -1):
    # This function is totally same as F.normalize(x, dim=-1)
    # avoid -inf
    return x / x.norm(p=2, dim=dim, keepdim=True)