# NIR Parameters Documentation

This document provides a comprehensive explanation of all parameters used in the NIR (Neural Image Registration) project for medical image registration.

## Training Parameters

### Core Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config_filepath` / `-c` | str | - | Path to YAML configuration file |
| `--reg_logging_root` | str | `./logs_HLN-12-1` | Root directory for saving model logs and checkpoints |
| `--experiment_name` | str | **Required** | Name of subdirectory in logging_root where summaries and checkpoints will be saved |
| `--batch_size` | int | 1 | Number of samples per training batch |
| `--lr` | float | 1e-4 | Learning rate for the optimizer |
| `--num_epochs` | int | 10000 | Total number of training epochs |

### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_type` | str | `sine` | Neural network architecture type. Options:<br/>- `sine`: All sine activations<br/>- `relu`: All ReLU activations<br/>- `nerf`: ReLU activations with positional encoding (NeRF style)<br/>- `rbf`: Input RBF layer, rest ReLU<br/>- `ode_gaussian_sine`: ODE-based with Gaussian positional encoding and sine activations<br/>- `ode_hash_relu`: ODE-based with hash encoding and ReLU activations |
| `--num_hidden_layers` | int | 3 | Number of hidden layers in the MLP |
| `--gaussian_scale` | float | 5 | Gaussian scale parameter for positional encoding |

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sample_mtd` | str | `random` | Sampling method for training data. Options:<br/>- `random`: Random sampling<br/>- `whole`: Whole volume sampling |
| `--sample_exp` | float | 18 | Number of sampling points is 2^sample_exp |

### Loss Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--loss_type` | str | `image_mse` | Loss function type. Options:<br/>- `image_mse`: Mean Squared Error loss<br/>- `l1`: L1 loss<br/>- `ncc`: Normalized Cross Correlation loss<br/>- `discrete_jacobian`: Jacobian determinant regularization<br/>- Combinations: `image_mse+discrete_jacobian` |

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--image_dir` | str | - | Directory containing 3D medical images |
| `--fixed_pid` | str | - | Patient ID of the fixed (target) image |
| `--moving_pid` | str | - | Patient ID of the moving (source) image |

### Logging and Checkpoint Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs_til_ckpt` | int | 25 | Number of epochs between model checkpoints |
| `--steps_til_summary` | int | 1000 | Number of steps between tensorboard summaries |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pretrained_dir` | str | - | Directory containing pretrained model weights |

### Available Sampling Methods

1. **Random Sampling** (`random`)
   - Randomly samples points from the entire volume
   - Number of points: 2^sample_exp
   - Good for general training but may miss important regions

2. **Whole Volume Sampling** (`whole`)
   - Samples the entire volume at once
   - Used for inference and evaluation
   - Most comprehensive but memory intensive

3. **Patch Sampling** (`patch`)
   - Samples random cubic patches from the volume
   - Patch size: 2^(sample_exp/3) in each dimension
   - Good for capturing local structures

4. **MAE Sampling** (`mae`)
   - Masked Autoencoder style sampling
   - Samples multiple random patches of fixed size (32x32x32)
   - Number of patches: 2^sample_exp / 2^15
   - Good for self-supervised learning

5. **Downsampled Random** (`downsize_random`)
   - Downsamples volume by factor of 3 with random jitter
   - Each point gets random offset within jitter range
   - Balances efficiency with coverage

6. **Downsampled Center** (`downsize_center`)
   - Downsamples volume by factor of 3 with center alignment
   - More structured than random downsampling
   - Consistent sampling pattern

7. **Downsampled Unified Random** (`downsize_unified_random`)
   - Downsamples with unified random jitter across all dimensions
   - Single random offset applied to entire volume
   - Maintains spatial coherence

8. **Combination Methods**:
   - `downsize_center+unified_random`: Combines center and unified random sampling
   - `downsize_unified_random+patch`: Combines unified random with patch sampling
   - `downsize_unified_random+mae`: Combines unified random with MAE sampling
   - `downsize_center+patch`: Combines center with patch sampling
   - `downsize_center+random`: Combines center with random sampling

9. **Specialized Methods**:
   - `downsize_unified_random_double`: Returns two samples for dual training
   - `downsize_random_syn`: Synthetic sampling using fixed data model
   - `downsize`: MRI-specific downsampling with mask

## Usage Examples

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_efficient_3d_ct_reg_pairwise.py \
    -c configs/ct_3d_pairwise_reg.yml \
    --batch_size 1 \
    --reg_logging_root ./logs \
    --experiment_name my_experiment \
    --num_hidden_layers 3 \
    --model_type sine \
    --gaussian_scale 5 \
    --sample_mtd random \
    --loss_type image_mse \
    --image_dir /path/to/images \
    --moving_pid patient_001 \
    --fixed_pid patient_002 \
    --lr 0.0002 \
    --num_epochs 1000
```

### Testing
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_efficient_3d_ct_reg_pairwise_oasis.py \
    --logging_root ./logs \
    --experiment_name my_experiment \
    --model_type sine \
    --gaussian_scale 5 \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --num_hidden_layers 3 \
    --moving_pid patient_001 \
    --fixed_pid patient_002 \
    --epoch_name epoch_1000
```

### Evaluation
```bash
python scripts/eval_3d_ct_reg_pairwise_oasis.py \
    --logging_root ./logs \
    --experiment_name my_experiment \
    --image_dir /path/to/images \
    --mask_dir /path/to/masks \
    --moving_pid patient_001 \
    --fixed_pid patient_002 \
    --epoch_name epoch_1000
```

## Tips and Best Practices

1. **Model Type Selection**:
   - Use `sine` for high-frequency details
   - Use `relu` for standard applications
   - Use `nerf` for better generalization

2. **Gaussian Scale**:
   - Higher values capture more high-frequency details
   - Lower values are more stable but may miss fine details

3. **Sampling**:
   - `random` sampling is faster but may miss important regions
   - `whole` sampling is more thorough but slower

4. **Loss Functions**:
   - Start with `image_mse` for basic registration
   - Add `discrete_jacobian` for topology preservation
   - Use `ncc` for better handling of intensity variations

5. **Learning Rate**:
   - Start with 1e-4 and adjust based on convergence
   - Lower learning rates (1e-5) for fine-tuning

6. **Sampling Method**:
    - **For initial training**: Use `random` or `downsize_random`
    - **For fine-tuning**: Use `patch` or `mae` for local details
    - **For inference**: Use `whole` for complete volume processing
    - **For memory efficiency**: Use any `downsize_*` method
    - **For multi-scale training**: Use combination methods like `downsize_center+patch`

