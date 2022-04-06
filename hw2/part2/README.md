# Image Classification Using CNN

## Preparation
1. Directory Structure
    ```bash
    .
    ├─p2_data
    | ├─annotations
    | | ├─public_test_annos.json
    | | └─train_annos.json
    | ├─public_test
    | | └─*.jpg
    | ├─train
    | | └─*.jpg
    | └─unlabeled
    |   └─*.jpg
    ├─eval.py
    ├─cfg.py
    ├─eval.py
    ├─main.py
    ├─myDatasets.py
    ├─myModels.py
    ├─tool.py
    └─requirements.txt
    ```
2. Create a virtual environment and install the required packages.
    ```bash
    conda create --name <env> --file requirements.txt
    ```
3. Define your own model or choose one from `myModels.py`.
4. Configuration setting in `cfg.py`
    ```python
    model_cfg = {
        'model_type': str,
        'data_root' : './p2_data/annotations/train_annos.json',
        
        # ratio of training images and validation images 
        'split_ratio': float,
        # set a random seed to get a fixed initialization 
        'seed': int,
        
        # training hyperparameters
        'batch_size': int,
        'lr': float,
        'milestones': List[int],
        'num_out': int,   # number of output classes
        'num_epoch': int,
    }
    ```
5. Import your model and configuration to `main.py` and `eval.py`.

## Training
```bash
python main.py
```

## Testing
```
python eval.py <--path Path> <--test_anno anno>
```
- `--path`: model_path. Default="./save_dir/Resnet50/best_model.pt"
- `--test`: annotaion for test image. Default="./p2_data/annotations/public_test_annos.json"
