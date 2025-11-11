import json
import logging
import os
import sys
import time

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from utils.DataLoader import get_feature_data, get_dataset
from utils.create_executor import create_executor
from utils.utils import get_parameter_sizes, to_var
from utils.EarlyStopping import EarlyStopping
from importlib import import_module
from evaluate_model import evaluate_model
from utils.utils import set_random_seed, save_model
import copy

def train_model(config):
    # get segment features
    data_feature = get_feature_data(config)

    model, loss_func, optimizer, start_epoch, start_run = create_executor(config)

    val_metric_all_runs, test_metric_all_runs,  = [], []

    try:
        for run in range(start_run, config['num_runs']):
            set_random_seed(seed=run)
            config['seed'] = run

            # get dataloader of full data
            dataloader = get_dataset(config, data_feature)

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{config['seed']}/", exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(
                f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{config['seed']}/{str(time.time())}.log")
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

            early_stopping = EarlyStopping(patience=config['patience'], save_model_folder=config['save_model_folder'],
                                           save_model_name=config['save_model_name'], logger=logger, model_name=config['model'],seed=config['seed'])

            if "metrics_train" not in config:
                config['metrics_train'] = config['metrics_eval']
            for epoch in range(start_epoch, config['num_epochs']):
                model.train()
                config['current_epoch'] = epoch

                # store train losses and metrics
                current_time = time.time()
                train_losses, train_metrics = [], []
                train_data_loader_tqdm = tqdm(dataloader['train'], ncols=120)

                for batch_idx, train_data in enumerate(train_data_loader_tqdm):
                    optimizer.zero_grad()
                    (features, truth_data) = train_data
                    features = to_var(features, config['device'])

                    outputs = model(features, config, data_feature)
                    loss = loss_func(truth=to_var(truth_data, config['device']), predict=outputs)

                    train_losses.append(loss.item())
                    train_metrics.append(getattr(import_module('utils.metrics'),
                                        config['metrics_train'])(preds=outputs, labels=truth_data))

                    loss.backward()
                    clip_grad_norm_(parameters=model.parameters(), max_norm=50, norm_type=2)

                    optimizer.step()
                    train_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'], loss_func=loss_func, config=config, data_feature= data_feature)

                logger.info(f'Epoch: {epoch + 1}, train and val duration per epoch: {(time.time() - current_time):.2f} seconds, '
                            f'learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')

                for metric_name in train_metrics[0].keys():
                    logger.info(
                        f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(
                        f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

                # perform testing once after test_interval_epochs
                if (epoch + 1) % config['test_interval_epochs'] == 0:
                    test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'], loss_func=loss_func, config=config, data_feature= data_feature)

                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    for metric_name in test_metrics[0].keys():
                        logger.info(
                            f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

                # select the best model based on all the validate metrics, higher_better: True
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    if ('recall' in metric_name or 'precision' in metric_name or 'accuracy' in metric_name or 'f1' in metric_name or
                            'top' in metric_name or 'auc' in metric_name) or (metric_name in ['PEARR']):
                        val_metric_indicator.append(
                            (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                    else:
                        val_metric_indicator.append(
                            (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), False))

                early_stop = early_stopping.step(val_metric_indicator, model)
                if early_stop:
                    break

                save_model(f"{config['save_model_folder']}/final_model.pkl",
                           **{'model_state_dict': copy.deepcopy(model.state_dict()),
                              'epoch': config['current_epoch'],
                              'run': config['seed'],
                              'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})

            start_epoch = 0
            # load the best model
            early_stopping.load_checkpoint(model)

            # evaluate the best model
            logger.info(f"get final performance on dataset {config['dataset']}...")

            val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'], loss_func=loss_func,
                                                       config=config, data_feature= data_feature)

            test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'], loss_func=loss_func,
                                                       config=config, data_feature= data_feature, mode='final_test')

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
            logger.info(f"Run {run + 1} after {config['current_epoch']} epochs cost {single_run_time:.2f} seconds.")

            val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            # avoid the overlap of logs
            if run < config['num_runs'] - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'] ,config['dataset'])

            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{config['save_model_name']}_seed{config['seed']}.json")

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
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
    finally:
        save_model(f"{config['save_model_folder']}/final_model.pkl",
                   **{'model_state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': config['current_epoch'],
                      'run': config['seed'],
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})

    sys.exit()
