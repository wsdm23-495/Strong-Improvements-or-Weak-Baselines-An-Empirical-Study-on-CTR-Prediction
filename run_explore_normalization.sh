# Explore the impact of normalization for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/norm_frappe_afn+.yaml --gpu 0 > logs/norm_frappe_afn+.log;
# Explore the impact of normalization for AutoInt+ on Frappe
nohup python -u run_param_tuner.py --config config/AutoInt/norm_frappe_autoint+.yaml --gpu 0 > logs/norm_frappe_autoint+.log;
# Explore the impact of normalization for AFN+ on Movielens
nohup python -u run_param_tuner.py --config config/AFN/norm_movielens_afn+.yaml --gpu 0 > logs/norm_movielens_afn+.log;
# Explore the impact of normalization for AutoInt+ on Movielens
nohup python -u run_param_tuner.py --config config/AutoInt/norm_movielens_autoint+.yaml --gpu 0 > logs/norm_movilens_autoint+.log;

