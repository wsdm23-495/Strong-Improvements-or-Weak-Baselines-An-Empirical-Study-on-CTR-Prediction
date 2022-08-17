# Explore the impact of earlystop for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/earlystop_frappe_afn+.yaml --gpu 0 > logs/earlystop_frappe_afn+.log;
# Explore the impact of earlystop for AFN+ on Movielens
nohup python -u run_param_tuner.py --config config/AFN/earlystop_movielens_afn+.yaml --gpu 0 > logs/earlystop_movielens_afn+.log;
