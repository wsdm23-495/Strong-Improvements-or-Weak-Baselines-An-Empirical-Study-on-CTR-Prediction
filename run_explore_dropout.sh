# Explore the impact of dropout for AFN+ on Criteo
nohup python -u run_param_tuner.py --config config/AFN/dropout_criteo_afn+.yaml --gpu 0 > logs/dropout_criteo_afn+.log;
# Explore the impact of dropout for AutoInt+ on Criteo
nohup python -u run_param_tuner.py --config config/AutoInt/dropout_criteo_autoint+.yaml --gpu 0 > logs/dropout_criteo_autoint+.log;
# Explore the impact of dropout for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/dropout_frappe_afn+.yaml --gpu 0 > logs/dropout_frappe_afn+.log;
# Explore the impact of dropout for AutoInt+ on Frappe
nohup python -u run_param_tuner.py --config config/AutoInt/dropout_frappe_autoint+.yaml --gpu 0 > logs/dropout_frappe_autoint+.log;



