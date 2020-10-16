#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import codecs
import random
import torch.nn as nn
from utils.logging import init_logger
from inputters.dataset import make_text_iterator_from_file
from inputters.dataset import build_dataset_iter, load_dataset, save_fields_to_vocab, load_fields
import onmt.transformer as nmt_model
import torch
import onmt.opts as opts
from onmt.reinforcor import build_reinforcor
from onmt.transformer import build_model
from utils.optimizers import build_optim
import sys

import data_loader
import os
def _check_save_model_path(opt):
  save_model_path = os.path.abspath(opt.save_model)
  model_dirname = os.path.dirname(save_model_path)
  if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)
def main(opt):
  if opt.gpu==0:
    device_id=0
  else :
    device_id=-1

  # dummy_parser = configargparse.ArgumentParser(description='reinforce.py')
  # opts.model_opts(dummy_parser)
  # dummy_opt = dummy_parser.parse_known_args([])[0]
  # # build the model and get the checkpoint and field
  # fields, model = nmt_model.load_reinforce_model(opt, dummy_opt.__dict__)
  opt = training_opt_reinforcing(opt, device_id)
  init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  # Load checkpoint if we resume from a previous training.
  if opt.train_from:
      logger.info('Loading checkpoint from %s' % opt.train_from)
      checkpoint = torch.load(opt.train_from,
                              map_location=lambda storage, loc: storage)

      # Load default opts values then overwrite it with opts from
      # the checkpoint. It's usefull in order to re-train a model
      # after adding a new option (not set in checkpoint)
      dummy_parser = configargparse.ArgumentParser()
      opts.model_opts(dummy_parser)
      default_opt = dummy_parser.parse_known_args([])[0]

      model_opt = default_opt
      model_opt.__dict__.update(checkpoint['opt'].__dict__)
  else:
     checkpoint=None
     model_opt=opt
  # Load fields generated from preprocess phase.

  fields = load_fields(opt, checkpoint)
  # Build model.
  model = build_model(model_opt, opt, fields, checkpoint)
  n_params, enc, dec = _tally_parameters(model)
  logger.info('encoder: %d' % enc)
  logger.info('decoder: %d' % dec)
  logger.info('* number of parameters: %d' % n_params)
  _check_save_model_path(opt)
  optim = build_optim(model, opt, checkpoint)
  optim.learning_rate=1e-5
  # Build model saver
  model_saver = build_model_saver(model_opt, opt, model, fields, optim)
  reinforcor = build_reinforcor(model,fields,opt,model_saver=model_saver,optim=optim)
  # out_file = codecs.open(opt.output, 'w+', 'utf-8')
  # X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader.test_mosei_emotion_data()
  # src_path=X_train
  # src_iter = make_text_iterator_from_file(src_path)#(opt.src)
  # tgt_path=y_train
  # tgt_iter=make_text_iterator_from_file(tgt_path)

  def train_iter_fct():
      return build_dataset_iter(load_dataset("train", opt), fields, opt)


  # if opt.tgt is not None:
  #   tgt_iter = make_text_iterator_from_file(opt.tgt)
  # else:
  #   tgt_iter = None
  # reinforcor.reinforce(src_data_iter=src_iter,
  #                      tgt_data_iter=tgt_iter,
  #                      batch_size=opt.batch_size,
  #                      out_file=out_file)
  reinforcor.reinforce(train_iter_fct,opt.rein_steps)



def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.save_checkpoint_steps,
                             opt.keep_checkpoint)
    return model_saver
class ModelSaver(object):

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def maybe_save(self, step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return

        chkpt, chkpt_name = self._save(step)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'optim': self.optim,
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        os.remove(name)

def training_opt_reinforcing(opt, device_id):

  if torch.cuda.is_available() and not opt.gpu_ranks:
    logger.info("WARNING: You have a CUDA device, \
                should run with -gpu_ranks")

  if opt.seed > 0:
    torch.manual_seed(opt.seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(opt.seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

  if device_id >= 0:
    torch.cuda.set_device(device_id)
    if opt.seed > 0:
      # These ensure same initialization in multi gpu mode
      torch.cuda.manual_seed(opt.seed)

  return opt
def _tally_parameters(model):
  n_params = sum([p.nelement() for p in model.parameters()])
  enc = 0
  dec = 0
  for name, param in model.named_parameters():
    if 'encoder' in name:
      enc += param.nelement()
    else:
      dec += param.nelement()
  return n_params, enc, dec

if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='reinforce.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  opts.config_opts(parser)
  opts.reinforce_opts(parser)

  opt = parser.parse_args()
  logger = init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  main(opt)
  sys.stdout.flush()
  input('[Press Any Key to start another run]')

