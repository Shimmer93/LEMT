# LEMT

## Dependency Installation
1. Create a conda environment and activate it:
    ```
    conda create -n lemt python=3.12
    conda activate lemt
    ```
2. Install Lightning:
    ```
    conda install lightning==2.5.5 -c conda-forge
    ```

2. Install dependencies in `requirements.txt` (or `requirements_cu111.txt`):
    ```
    pip install -r requirements.txt
    ```
3. Install pointnet2 for P4Transformer:
    ```
    cd model/P4Transformer
    python setup.py install
    ```

## Data Download and Preprocessing
### Raw Data
To be completed
### Preprocessed Data
To be completed

## Training and Testing
### Training
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name mmfi_p4t -c cfg/base/mmfi.yml --version $(date +'%Y%m%d_%H%M%S')
```
### Testing
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name mmfi_p4t -c cfg/base/mmfi.yml --version $(date +'%Y%m%d_%H%M%S') --checkpoint_path logs/mmfi_p4t/20251006_020302/P4Transformer-epoch=93-val_mpjpe=0.0764.ckpt --test
```
