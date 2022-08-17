# Explore the impact of regularization for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/reg_frappe_afn+.yaml --gpu 0 > logs/reg_frappe_afn+.log;
# Explore the impact of regularization for AFN+ on Movielens
nohup python -u run_param_tuner.py --config config/AFN/reg_movielens_afn+.yaml --gpu 0 > logs/reg_movielens_afn+.log;
