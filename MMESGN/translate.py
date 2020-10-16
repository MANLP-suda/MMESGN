#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import codecs

from utils.logging import init_logger
from inputters.dataset import make_text_iterator_from_file
import onmt.opts as opts
from onmt.translator import build_translator
import data_loader
import os
def main(opt,model_path):

  translator = build_translator(opt,model_path)
  out_file = codecs.open(opt.output, 'w+', 'utf-8')
  # X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader.test_mosei_emotion_data()
  X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader.read_cmumosei_emotion_pkl()
  src_path=X_test
  src_iter = make_text_iterator_from_file(src_path)#(opt.src)
  tgt_path=y_test
  tgt_iter=make_text_iterator_from_file(tgt_path)
  # if opt.tgt is not None:
  #   tgt_iter = make_text_iterator_from_file(opt.tgt)
  # else:
  #   tgt_iter = None
  translator.translate(src_data_iter=src_iter,
                       tgt_data_iter=tgt_iter,
                       batch_size=opt.batch_size,
                       out_file=out_file)
  out_file.close()


if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='translate.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  opts.config_opts(parser)
  opts.translate_opts(parser)

  opt = parser.parse_args()
  logger = init_logger(opt.log_file)
  logger.info("Input args: %r", opt)
  path='rein_model/rein_model_step'
  for i in range(0,25000,10):
    current_path=path+'_'+str(i)+'.pt'
    if os.path.exists(current_path):

      model_path=current_path
      opt.output='rein_data/rein.tran'+'_'+str(i)
      main(opt,model_path)
    else :

      continue
