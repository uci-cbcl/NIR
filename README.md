# NIR: Medical Image Registration via Neural Fields

[![arXiv](https://img.shields.io/badge/arXiv-2206.03111-b31b1b.svg)](https://arxiv.org/abs/2206.03111)
[![MedIA](https://img.shields.io/badge/MedIA-2024.103249-green)](https://www.sciencedirect.com/science/article/pii/S1361841524001749)

> **Medical Image Registration via Neural Fields**
> 
> [Shanlin Sun]([https://siwensun.github.io/]), [Kun Han](), [Chenyu You](), [Hao Tang](), [Deying Long](), [Junayed Naushad](), [Xiangyi Yan](), [Haoyu Ma](), [Pooya Khasravi](), [James S. Duncan]() and [Xiaohui Xie]([https://ics.uci.edu/~xhx/])
> 
> - **Institutions**: University of California, Irvine; Yale University
> - **Contact**: [Shanlin Sun](https://siwensun.github.io/) (shanlins@uci.edu)

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU

### Setup

```bash
# Clone the repository
git clone https://github.com/uci-cbcl/NIR
cd NIR

# Create conda environment
conda env create -f environment.yml

# Activate conda environment
conda activate nir
```

## Usage

### Training

Train the NIR model for 3D CT image registration:
```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train_efficient_3d_ct_reg_pairwise.py \
    -c ${config_file} \
    --batch_size ${batch_size} \
    --reg_logging_root ${reg_logging_root} \
    --experiment_name ${experiment_name} \
    --num_hidden_layers ${num_hidden_layers} \
    --model_type ${model_type} \
    --gaussian_scale ${gaussian_scale} \
    --sample_mtd ${sample_mtd} \
    --loss_type ${loss_type} \
    --image_dir ${image_dir} \
    --moving_pid ${moving_pid} \
    --fixed_pid ${fixed_pid} \
    --lr 0.0002 \
    --num_epochs ${num_epochs} \
    --pretrained_dir ${pretrained_dir} 
```

The training process can be configured using YAML files in the `configs/` directory:

- `ct_3d_pairwise_reg.yml`: Configuration for 3D CT pairwise registration

Example configuration parameters:
- `num_epochs`: Number of training epochs
- `lr`: Learning rate
- `model_type`: Neural network architecture type
- `num_hidden_layers`: Number of hidden layers
- `sample_mtd`: Sampling method for training

### Inference

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/test_3d_ct_reg_pairwise_oasis.py\
    --logging_root ${reg_logging_root} \
    --experiment_name ${experiment_name} \
    --model_type ${model_type} \
    --gaussian_scale ${gaussian_scale} \
    --image_dir ${image_dir} \
    --mask_dir ${mask_dir} \
    --num_hidden_layers ${num_hidden_layers} \
    --moving_pid ${moving_pid} \
    --fixed_pid ${fixed_pid} \
    --epoch_name ${epoch_name} 
```

### Evaluation

```bash
python scripts/eval_3d_ct_reg_pairwise_oasis.py \
    --logging_root ${reg_logging_root} \
    --experiment_name ${experiment_name} \
    --image_dir ${image_dir} \
    --mask_dir ${mask_dir} \
    --moving_pid ${moving_pid} \
    --fixed_pid ${fixed_pid} \
    --epoch_name ${epoch_name}
```

## Documentation

- **[Parameter Reference](PARAMETERS.md)**: Comprehensive documentation of all script parameters, configuration options, and usage examples.


## Project Structure

```
NIR_release/
├── configs/                    # Configuration files
│   └── ct_3d_pairwise_reg.yml # Pairwise registration config
├── README.md                   # This file
└── train_efficient_3d_ct_reg_pairwise.py  # Main training script
```

## Citation
```bib
@article{sun2024medical,
  title={Medical image registration via neural fields},
  author={Sun, Shanlin and Han, Kun and You, Chenyu and Tang, Hao and Kong, Deying and Naushad, Junayed and Yan, Xiangyi and Ma, Haoyu and Khosravi, Pooya and Duncan, James S and others},
  journal={Medical Image Analysis},
  volume={97},
  pages={103249},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See the `LICENSE` file for details or visit `https://creativecommons.org/licenses/by/4.0/`.
