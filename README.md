# MMESGN  Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection
Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection
This is the code for our paper *MMESGN: Transformer-based Label Set Generation for Multi-modal Multi-label Emotion Detection* [[pdf]](https://dl.acm.org/doi/10.1145/3394171.3413577)

***********************************************************
# CODE
## The code has been released,  you can email me if you have problems

***********************************************************

## preprocess data：Modify the data path in data_loader.py first,then:
bash preprocess.sh
## train
bash train.sh
## reinforcement ： select one of train checkpoints and copy the path to -train_from in reinforce.sh,then
bash reforce.sh
## translate
bash translate.sh
## test score
python3 rein_evaluate_all.py


## Requirements
* Ubuntu 16.0.4
* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.1.0

## Dataset
Our used Mosei dataset can be downloaded from the page [this link](https://github.com/A2Zadeh/CMU-MultimodalSDK). The preprocess of the raw data clearly published in the information page. Download the data and follow the usage in readme.md

## Future 
The code will be published after the ACM MM conference, happy to see you reading my code in the future.
