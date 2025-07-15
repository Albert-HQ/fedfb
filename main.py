import sys, os
working_dir = r'C:\Users\10733\Downloads\fedfb\fedfb'
sys.path.insert(1, os.path.join(working_dir, 'FedFB'))
os.environ["PYTHONPATH"] = os.path.join(working_dir, 'FedFB')

from DP_run import sim_dp, sim_dp_man

sim_dp_man(
    method        ='fedfb',
    model         ='multilayer perceptron',
    dataset       ='compas',
    num_sim       =1,
    seed          =24,
    num_rounds    =10,      # ← 通信轮数
    local_epochs  =2,
    learning_rate =0.001,
    alpha         =1      # 公平性权重 λ
)