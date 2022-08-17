# Explore the impact of learning rate for AFN on Frappe
nohup python -u run_param_tuner.py --config config/AFN/lr_frappe_afn.yaml --gpu 0 > logs/lr_frappe_afn.log;
# Explore the impact of learning rate for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/lr_frappe_afn+.yaml --gpu 0 > logs/lr_frappe_afn+.log;
# Explore the impact of learning rate for AFN on Criteo
nohup python -u run_param_tuner.py --config config/AFN/lr_criteo_afn.yaml --gpu 0 > logs/lr_criteo_afn.log;
# Explore the impact of learning rate for AFN+ on Criteo
nohup python -u run_param_tuner.py --config config/AFN/lr_criteo_afn+.yaml --gpu 0 > logs/lr_criteo_afn+.log;