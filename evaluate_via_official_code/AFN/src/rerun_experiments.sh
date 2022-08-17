#!/bin/bash

filename="raw_AFN";

# run AFN+
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --hidden_size=600 --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609 > logs/rerun/rerun_frappe_afn+.log;
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --hidden_size=800 --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859 >logs/rerun/rerun_movielens_afn+.log;
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --hidden_size=1500 --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617 > logs/rerun/rerun_criteo_afn+.log;


## run AFN
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --ensemble=False --hidden_size=600 --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609 > logs/rerun/rerun_frappe_afn.log;
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --ensemble=False --hidden_size=800 --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859 > logs/rerun/rerun_movielens_afn.log;
CUDA_VISIBLE_DEVICES=0 nohup python -u ./${filename}.py --ensemble=False --hidden_size=1500 --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617 > logs/rerun/rerun_criteo_afn.log;


