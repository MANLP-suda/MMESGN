# coding:utf-8
'''
for loading the CS data
'''
import sys
# print(sys.path)
# sys.path.append('g:\\CMU-MultimodalSDK-master\\mmsdk\\')

import pickle as pkl
import emo_preprocess
# from keras.utils import np_utils


def read_cmumosei_emotion_pkl():
  import pickle
    #read picke data
  dataset_path="~/mosei_emotion_aligned_60.pkl"
  dataset = pickle.load(open(dataset_path, 'rb'))#dict
  X_train=[dataset['train']['text'] ,dataset['train']['vision'] ,dataset['train']['audio'] ]
  X_valid=[dataset['valid']['text'] ,dataset['valid']['vision'] ,dataset['valid']['audio'] ]
  X_test=[dataset['test']['text'] ,dataset['test']['vision'] ,dataset['test']['audio'] ]
  y_train=dataset['train']['labels'] 
  y_valid=dataset['valid']['labels'] 
  y_test=dataset['test']['labels'] 

  x_train = emo_preprocess.preprocess_data(X_train)
  x_valid = emo_preprocess.preprocess_data(X_valid)
  x_test = emo_preprocess.preprocess_data(X_test)

  y_train_emo=emo_preprocess.preprocess_emo_2(y_train)
  y_valid_emo=emo_preprocess.preprocess_emo_2(y_valid)
  y_test_emo=emo_preprocess.preprocess_emo_2(y_test)
  return x_train, x_valid, x_test, y_train_emo, y_valid_emo, y_test_emo
