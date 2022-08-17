# tuner_config.yaml:
# gpu_id: 0
# nohup python -u run_param_tuner.py --config config/{Model}/order_{Dataset}_{Model}.yaml --gpu {gpu_id} > logs/order_{Dataset}_{Model}.log &


# Explore the impact of order for AutoInt+ on Frappe
nohup python -u run_param_tuner.py --config config/AutoInt/order_frappe_autoint+.yaml --gpu 0 > logs/order_frappe_autoint+.log &
# Explore the impact of order for DCN on Frappe
nohup python -u run_param_tuner.py --config config/DCN/order_frappe_dcn.yaml --gpu 0 > logs/order_frappe_dcn.log &
# Explore the impact of order for xDeepFM on Frappe
nohup python -u run_param_tuner.py --config config/xDeepFM/order_frappe_xdeepfm.yaml --gpu 0 > logs/order_frappe_xdeepfm.log &

# Explore the impact of order for AutoInt+ on Movielens
nohup python -u run_param_tuner.py --config config/AutoInt/order_movielens_autoint+.yaml --gpu 0 > logs/order_movilens_autoint+.log &
# Explore the impact of order for DCN on Frappe
nohup python -u run_param_tuner.py --config config/DCN/order_movielens_dcn.yaml --gpu 0 > logs/order_movielens_dcn.log &
# Explore the impact of order for xDeepFM on Frappe
nohup python -u run_param_tuner.py --config config/xDeepFM/order_movielens_xdeepfm.yaml --gpu 0 > logs/order_movielens_xdeepfm.log &
