# Explore the impact of DNN Layers for AFN+ on Frappe
nohup python -u run_param_tuner.py --config config/AFN/dnn_layer_frappe_afn+.yaml --gpu 0 > logs/dnn_layer_frappe_afn+.log &
# Explore the impact of DNN Layers for AFN+ on Movielens
nohup python -u run_param_tuner.py --config config/AFN/dnn_layer_movielens_afn+.yaml --gpu 0 > logs/dnn_layer_movielens_afn+.log &
# Explore the impact of DNN Layers for AFN+ on Criteo
nohup python -u run_param_tuner.py --config config/AFN/dnn_layer_criteo_afn+.yaml --gpu 0 > logs/dnn_layer_criteo_afn+.log &
