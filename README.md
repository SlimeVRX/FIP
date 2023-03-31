##  SMPL body model
1. Download SMPL model from [here](https://smpl.is.tue.mpg.de/). 
2. Put the model file in pkl format into the models folder.

## Prepare test datasets (optional)

1. Download DIP-IMU dataset from [here](https://dip.is.tue.mpg.de/).
2. Unzip the DIP dataset and put it into data/dataset_work/DIP_IMU. The directory opens the file for processing DIP_IMU dataset in the preprocess_dataset.py file.

## evaluate
You can execute the following code to get the results in the paper.

```python
python evaluate.py weights.tar 0.6
```

## train

If you want to retrain, can you download the AMASS[here](https://amass.is.tue.mpg.de/) dataset and train with the following code?

```python
python train.py -c -b 200 --epochs 300 --posenet --save-dir save_pose  --lr 5e-4 # pretrain

python train.py -c -b 200 --epochs 300 --posenet --save-dir save_pose  --lr 5e-4 -f # fineturne
```

## live demo

Run live demo program
```python
python live_demo.py
```