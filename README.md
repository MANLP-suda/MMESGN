# MMESGN  Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection
Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection
This is the code for our paper *MMESGN: Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection* [[pdf]](https://dl.acm.org/doi/10.1145/3394171.3413577)
<img src="https://user-images.githubusercontent.com/69071185/146308352-2c7766b8-c6a7-4028-9066-14c6cdeb9909.png" width="700">

***********************************************************
# CODE
## The code has been released,  you can email me if you have problems

***********************************************************

## preprocess data

Modify the data path in data_loader.py. Then:
```
bash preprocess.sh
```
## train

Training the based model by:
```
bash train.sh
```

## reinforcement 
We utilize a reinforecement method to help optimize the model.
Select one of train checkpoints and copy the path to -train_from in reinforce.sh,then :
```
bash reforce.sh
```
## translate
Similar to translation in NMT model, we need to generate the emotion label set.
```
bash translate.sh
```
## test score
```
python3 rein_evaluate_all.py
```

## Requirements
* Ubuntu 16.0.4
* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.1.0

## Dataset
Our used Mosei dataset can be downloaded from the page [this link](https://github.com/A2Zadeh/CMU-MultimodalSDK). The preprocess of the raw data clearly published in the information page. Download the data and follow the usage in readme.md

## Future 
The code will be published after the ACM MM conference, happy to see you reading my code in the future.
