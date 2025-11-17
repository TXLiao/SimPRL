
Source code of the proposed method SimPRL in the paper "SimPRL: A Simple Contrastive Learning for Path Representation Learning by Joint GPS Trajectories and Road Paths" in IEEE TITS.

If there are any questions, please start a question in Github. Or just contact the author email (if the response has delays). 
## Installation

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Download the Dataset and Model weights
Please take the datasets (folder name`./processed_data`) and model weights (folder name `./repository`) from the [Google Drive](https://drive.google.com/file/d/1mWk4D33Q59XJYwVAtP3ZAgR_Ni6eYR0b/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/18s3Sqrip7o4jabvnGZzKRA?pwd=icrx) (password：icrx) into the root directory of the project.

Note: The datasets are sourced from publicly available datasets and is intended solely for research and educational purposes. If there are any copyright issues or concerns, please contact the author for removal.


## Project Structure

- `config`: Main configuration file for tasks and models.
- `processed_data/`: Directory containing preprocessed datasets for training and evaluation.
- - `dataset/`
- - -  `pretrain/`
- - -  `finetune/`
- - - `.pkl` files of node and edge feature generated from the OpenStreetMap for different datasets.
- `repository/`: Directory for storing model weights and checkpoints.
- -  `saved_model/`
- -  `saved_results/`
- `utils/`: Utility scripts for configuration parsing, data loading, and other helper functions.
- `logs/`: Folder for storing log files generated during training and evaluation.
- `models/`: Implementation of model architectures used in the project.
- `main.py`: Entry point for the project.

## Run

For pre-training
   ```bash
python .\main.py --task rrl_pretrain --model SimPRL --mode train --dataset chengdu  
   ```
For fine-tuning
   ```bash
#  Travel time estimation task
python .\main.py --task reg_finetune --model MLP_TTE --mode train --dataset chengdu  
#  Road segment classification task
python .\main.py --task reg_finetune --model MLP_CLS --mode train --dataset chengdu  
   ```

## Citation
```
@ARTICLE{SimPRL,
  author={Liao, Tianxi and Ta, Xuxiang and Xu, Yi and Han, Liangzhe and Sun, Leilei and Lv, Weifeng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={SimPRL: A Simple Contrastive Learning for Path Representation Learning by Joint GPS Trajectories and Road Paths}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Trajectory;Roads;Global Positioning System;Contrastive learning;Representation learning;Vectors;Training;Estimation;Imputation;Transforms;Data-based approaches;smart cities;road transportation;self-supervised representation learning},
  doi={10.1109/TITS.2025.3629800}}

```

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [TQDM](https://github.com/tqdm/tqdm)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Fast Map Matching](https://fmm-wiki.github.io/)
- [Chengdu Dataset](https://challenge.datacastle.cn/v3/cmptDetail.html?id=175)
- [Porto Dataset](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)
- [Trajectory Data Cleaning Pipeline](https://www.kaggle.com/code/mrganger/identifying-invalid-gps-points-in-taxi-trips)
