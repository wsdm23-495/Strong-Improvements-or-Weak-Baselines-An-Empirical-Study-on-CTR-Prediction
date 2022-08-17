# Explore the impact of intializer for AFN on Criteo
nohup python -u run_param_tuner.py --config config/AFN/initializer_criteo_afn.yaml --gpu 0 > logs/intializer_criteo_afn.log;
# Explore the impact of intializer for AFN+ on Criteo
nohup python -u run_param_tuner.py --config config/AFN/initializer_criteo_afn+.yaml --gpu 0 > logs/intializer_criteo_afn+.log;
# Explore the impact of intializer for AFN on Frappe
nohup python -u run_param_tuner.py --config config/AFN/initializer_frappe_afn.yaml --gpu 0 > logs/intializer_frappe_afn+.log;
# Explore the impact of intializer for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/initializer_frappe_afn+.yaml --gpu 0 > logs/intializer_frappe_afn+.log;
