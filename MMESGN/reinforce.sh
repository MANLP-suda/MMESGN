CUDA_VISIBLE_DEVICES=0 nohup python3 -u reinforce.py -batch_size 16 --accum_count 4 -train_from save_model/model_step_100.pt -data ./save_data/modal   -save_model ./rein_model/rein_model -optim adam -seed 1234 -rein_steps 2000 -gpu_ranks 0 -gpu 0 1>rein_model/rein.log 2>&1