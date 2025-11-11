import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import os
import json
from tqdm import tqdm
import numpy as np
from importlib import import_module
from utils.DataLoader import get_feature_data, get_dataset
from utils.create_executor import create_executor
from utils.utils import get_parameter_sizes, to_var
from utils.utils import set_random_seed, flatten_list

def evaluate_model(model: nn.Module, evaluate_data_loader: DataLoader, loss_func: nn.Module, config, data_feature, mode = 'val'):
    model.eval()
    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120)
        truths = []
        predicts = []

        for batch_idx, evaluate_data in enumerate(evaluate_data_loader_tqdm):
            (features, truth_data) = evaluate_data

            features = to_var(features, config['device'])

            outputs = model(features, config, data_feature)
            loss = loss_func(truth=to_var(truth_data, config['device']), predict=outputs)

            evaluate_losses.append(loss.item())

            evaluate_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')
            if config['task'] == 'reg_finetune':
                evaluate_metrics.append(getattr(import_module('utils.metrics'),
                                            config['metrics_eval'])(preds=outputs, labels=truth_data))
            elif config['task'] == 'cls_finetune':
                truth_data = truth_data.reshape(-1)
                label_index = torch.where(truth_data != -100)[0]
                outputs = outputs.cpu().detach()
                predicts.append(outputs[label_index])
                truths.append(truth_data[label_index])

        if config['task'] == 'cls_finetune':
            evaluate_metrics.append(getattr(import_module('utils.metrics'), config['metrics_eval'])(preds=predicts, labels=truths))

    return evaluate_losses, evaluate_metrics

def test_model(config):
    # get segment features
    data_feature = get_feature_data(config)

    # get dataloader of full data
    dataloader = get_dataset(config, data_feature)

    val_metric_all_runs, test_metric_all_runs, = [], []

    for run in range(0, config['num_runs']):
        set_random_seed(seed=run)
        seed = run
        model, loss_func, _, _, _ = create_executor(config, seed=seed)
        model.eval()

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/",
            exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {config.convert_all_config_to_strs()}')

        logger.info(f'model -> {model}')
        logger.info(f"model name: {config['model']}, #parameters: {get_parameter_sizes(model) * 4} B, "
                    f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

        # evaluate the best model
        logger.info(f"get final performance on dataset {config['dataset']}...")

        val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'],
                                                 loss_func=loss_func,
                                                 config=config, data_feature=data_feature)

        test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'],
                                                   loss_func=loss_func,
                                                   config=config, data_feature=data_feature, mode='final_test')

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < config['num_runs'] - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                 val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                             test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)

        # save_result_folder = f"./saved_results/{args.model_name}_{args.identify}/{args.dataset_name}"
        save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'],
                                          config['dataset'])
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{config['save_model_name']}_seed{seed}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {config['num_runs']} runs:")

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(
            f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(
            f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(
            f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
            f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

def evaluate_pretrain_model(model: nn.Module, evaluate_data_loader: DataLoader, loss_func: nn.Module, config, data_feature, mode = 'val'):
    model.eval()
    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120)

        for batch_idx, evaluate_data in enumerate(evaluate_data_loader_tqdm):
            (features, truth_data) = evaluate_data
            features = to_var(features, config['device'])
            # SimPRL
            loss, values, losses = model(features, config, data_feature)

            evaluate_losses.append(loss.item())
            # SimPRL
            evaluate_metrics.append(getattr(import_module('utils.metrics'),
                                         config['metrics_eval'])(values=values))

            evaluate_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics

def test_pretrain_model(config):
    # get segment features
    data_feature = get_feature_data(config)

    val_metric_all_runs, test_metric_all_runs, = [], []

    for run in range(0, config['num_runs']):
        set_random_seed(seed=run)
        seed = run

        # get dataloader of full data
        dataloader = get_dataset(config, data_feature)

        model, loss_func, _, _, _ = create_executor(config)
        model.eval()

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/",
            exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {config.convert_all_config_to_strs()}')

        logger.info(f'model -> {model}')
        logger.info(f"model name: {config['model']}, #parameters: {get_parameter_sizes(model) * 4} B, "
                    f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

        # evaluate the best model
        logger.info(f"get final performance on dataset {config['dataset']}...")

        val_losses, val_metrics = evaluate_pretrain_model(model=model, evaluate_data_loader=dataloader['pretrain_eval'],
                                                          loss_func=loss_func,
                                                          config=config, data_feature=data_feature)
        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            if metric_name == 'z1' or metric_name == 'z2':
                continue
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        # test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < config['num_runs'] - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                 val_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'],
                                          config['dataset'])
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{config['save_model_name']}_seed{seed}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        np.save(f"{save_result_folder}/{config['save_model_name']}_seed{seed}_val_metrics.npy",val_metrics)
        # np.save(f"{save_result_folder}/{config['save_model_name']}_seed{seed}_test_metrics.npy",test_metrics)

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {config['num_runs']} runs:")

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(
            f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(
            f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

if __name__ == '__main__':
    pass
