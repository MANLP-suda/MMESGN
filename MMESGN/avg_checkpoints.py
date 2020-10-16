from sys import argv
import torch
import glob
import re
import numpy as np


def get_checkpoints(path):
  #get checkpints from model save dir
  path_list = glob.glob(path + '/' + '*.pt')
  return path_list

def sort_checkpoints(path_list, start_iter=None, end_iter=None):
  if start_iter is None:
    start_iter = 0
  if end_iter is None:
    end_iter = 999999
    
  iter_list = []
  saved_models = []
  for path in path_list:
    # path: model_step_100000.pt
    p = re.compile(r'model_step_(.*)\.pt$')
    matches = p.findall(path)
    if len(matches) > 0:
      i = int(matches[0])
      if i >= start_iter and i <= end_iter:
        saved_models.append(path)
        iter_list.append(i)
  sorted_index = np.argsort(iter_list)
  path_list = [saved_models[i] for i in sorted_index]
  return path_list


if __name__=="__main__":
  start_iter = None
  end_iter = None
  number = None
  
  if len(argv) == 4:
    script, model_save_dir, ensemble_path, number = argv
  elif len(argv) == 5:
    script, model_save_dir, ensemble_path, start_iter, end_iter = argv
  elif len(argv) == 6:
    script, model_save_dir, ensemble_path, start_iter, end_iter, number = argv
  else:
    print("ERROR INPUT")
    exit(1)
    
  if model_save_dir is None:
    print("model_save_dir error")

  checkpoints_list = get_checkpoints(model_save_dir)
  checkpoints_list = sort_checkpoints(checkpoints_list, start_iter=start_iter, end_iter=end_iter)
  if number is None:
    number = len(checkpoints_list)
  checkpoints_list = checkpoints_list[-int(number):]
  print("Averaging checkpoints: \n{}".format(checkpoints_list))
 
  print("start average the last {} models".format(number)) 
  model_list = []
  generator_list = []
  for checkpoint_path in checkpoints_list:
    checkpoint = torch.load(checkpoint_path)
    model_list.append(checkpoint['model'])
    generator_list.append(checkpoint['generator'])
    vocab = checkpoint['vocab']
    opt = checkpoint['opt']
    optim = checkpoint['optim']
  
  model_dict = {}
  generator_dict = {}
  for model, generator in zip(model_list, generator_list):
    for key, value in model.items():
      value_sum = model_dict.get(key, 0)
      value_sum += value
      model_dict[key] = value_sum
    for key, value in generator.items():
      value_sum = generator_dict.get(key, 0)
      value_sum += value
      generator_dict[key] = value_sum
  
  model_dict_avg = {}
  generator_dict_avg = {}
  for key, value in model_dict.items():
    model_dict_avg[key] = value / int(number)
  for key, value in generator_dict.items():
    generator_dict_avg[key] = value / int(number)
  
  checkpoint_ensemble = {'vocab':vocab, 'opt': opt, 'model': model_dict_avg, 'generator':generator_dict_avg, 'optim':optim}
  torch.save(checkpoint_ensemble, ensemble_path)
  print("ensemble end and save model to {}".format(ensemble_path))