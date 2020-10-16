CUDA_VISIBLE_DEVICES=3  python3  translate.py -model save_model/model_step_1400_0.5458_kl.pt -src test.en -output rein_data/rein.tran -gpu 0 
 1>rein_data/tran.log 2>&1