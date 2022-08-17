#!/bin/bash

filename="our_AFN";

# run AFN+
#### Frappe
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --hidden_size=800 --init_type=xavier_norm --init_param=0.0001 --learning_rate=0.001 --l2_reg=0.001 --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609;
#### Movielens
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --hidden_size=600 --init_type=xavier_uniform --init_param=1e-05 --learning_rate=0.001 --l2_reg=0.0001 --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859;
#### Criteo
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --hidden_size=1500 --init_type=norm --init_param=0.0001 --learning_rate=0.0005 --l2_reg=1e-5 --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617;


# run AFN
#### Frappe
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --ensemble=False --hidden_size=1500 --init_type=xavier_norm --init_param=0.0001 --learning_rate=0.001 --l2_reg=0.0001  --data_dir=../data/frappe/ --model_dir=./checkpoint/frappe/ --field_size=10 --feature_size=5500 --instance_size=288609;
#### Movielens
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --ensemble=False --hidden_size=1000 --init_type=xavier_norm --init_param=1e-05 --learning_rate=0.001 --l2_reg=0.0001  --data_dir=../data/movielens/ --model_dir=./checkpoint/movielens/ --field_size=3 --feature_size=92000 --instance_size=2006859;
#### Criteo
CUDA_VISIBLE_DEVICES=0 python -u ./${filename}.py --ensemble=False --hidden_size=1500 --init_type=xavier_norm --init_param=0.0001 --learning_rate=0.0005 --l2_reg=1e-5  --data_dir=../data/criteo/ --model_dir=./checkpoint/criteo/ --field_size=39 --feature_size=2100000 --instance_size=45840617;

