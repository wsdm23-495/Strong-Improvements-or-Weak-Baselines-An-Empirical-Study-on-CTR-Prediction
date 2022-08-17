# Explore the impact of batch size for AFN on Frappe
nohup python -u run_param_tuner.py --config config/AFN/batchsize_frappe_afn.yaml --gpu 0 > logs/batchsize_frappe_afn.log;
# Explore the impact of batch size for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/batchsize_frappe_afn+.yaml --gpu 0 > logs/batchsize_frappe_afn+.log;
# Explore the impact of batch size for AFN on Criteo
nohup python -u run_param_tuner.py --config config/AFN/batchsize_criteo_afn.yaml --gpu 0 > logs/batchsize_criteo_afn.log;
# Explore the impact of batch size for AFN+ on Criteo
nohup python -u run_param_tuner.py --config config/AFN/batchsize_criteo_afn+.yaml --gpu 0 > logs/batchsize_criteo_afn+.log;