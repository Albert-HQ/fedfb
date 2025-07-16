import numpy as np
from utils import *
import torch
import random

# synthetic
def dataSplit(train_data, test_data, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    if Z == 2:
        z1_idx = train_data[train_data.z == 1].index
        z0_idx = train_data[train_data.z == 0].index

        client1_idx = np.concatenate((z1_idx[:int(client_split[0][0]*len(z1_idx))], z0_idx[:int(client_split[0][1]*len(z0_idx))]))
        client2_idx = np.concatenate((z1_idx[int(client_split[0][0]*len(z1_idx)):int((client_split[0][0] + client_split[1][0])*len(z1_idx))],
                                      z0_idx[int(client_split[0][1]*len(z0_idx)):int((client_split[0][1] + client_split[1][1])*len(z0_idx))]))
        client3_idx = np.concatenate((z1_idx[int((client_split[0][0] + client_split[1][0])*len(z1_idx)):], z0_idx[int((client_split[0][1] + client_split[1][1])*len(z0_idx)):]))
        random.shuffle(client1_idx)
        random.shuffle(client2_idx)
        random.shuffle(client3_idx)

        clients_idx = [client1_idx, client2_idx, client3_idx]

    elif Z == 3:
        z_idx, z_len = [], []
        for z in range(3):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

        clients_idx = []
        a, b = np.zeros(3), np.zeros(3)
        for c in range(4):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
            b += np.array(client_split[c]) * z_len
            clients_idx.append(np.concatenate((z_idx[0][int(a[0]):int(b[0])],
                                               z_idx[1][int(a[1]):int(b[1])],
                                               z_idx[2][int(a[2]):int(b[2])])))
            random.shuffle(clients_idx[c])

    train_dataset = LoadData(train_data, "y", "z")
    test_dataset = LoadData(test_data, "y", "z")

    synthetic_info = [train_dataset, test_dataset, clients_idx]
    return synthetic_info

def dataGenerate(seed = 432, train_samples = 3000, test_samples = 500,
                y_mean = 0.6, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    """
    Generate dataset consisting of two sensitive groups.
    """
    ########################
    # Z = 2:
    # 3 clients:
    #           client 1: %50 z = 1, %20 z = 0
    #           client 2: %30 z = 1, %40 z = 0
    #           client 3: %20 z = 1, %40 z = 0
    ########################
    # 4 clients:
    #           client 1: 50% z = 0, 10% z = 1, 20% z = 2
    #           client 2: 30% z = 0, 30% z = 1, 30% z = 2
    #           client 3: 10% z = 0, 30% z = 1, 30% z = 2
    #           client 4: 10% z = 0, 30% z = 1, 20% z = 2
    ########################
    np.random.seed(seed)
    random.seed(seed)

    train_data, test_data = dataSample(train_samples, test_samples, y_mean, Z)
    return dataSplit(train_data, test_data, client_split, Z)

synthetic_info = dataGenerate(seed = 123, test_samples = 1500, train_samples = 3500)

# Adult
sensitive_attributes = ['sex']
categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week',
            'native-country', 'salary']
label_name = 'salary'

adult = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
test['native-country_ Holand-Netherlands'] = 0
test = test[adult.columns]

# Adult - 使用狄利克雷分布进行客户端划分（确保最小样本数）
np.random.seed(1)
adult_private_idx = adult[adult['workclass_ Private'] == 1].index
adult_others_idx = adult[adult['workclass_ Private'] == 0].index
adult_mean_sensitive = adult['z'].mean()

# 狄利克雷分布参数设置
NUM_CLIENTS = 5  # 客户端数量
ALPHA = 0.1  # 狄利克雷分布参数 (越小越不均匀)
MIN_SAMPLES = 100  # 每个客户端的最小样本数
MIN_SAMPLES_PER_GROUP = 20  # 每个客户端每个敏感属性组的最小样本数

# 获取不同敏感属性的索引
adult_z0_idx = adult[adult['z'] == 0].index.tolist()
adult_z1_idx = adult[adult['z'] == 1].index.tolist()

print(f"\n数据集信息:")
print(f"总样本数: {len(adult)}")
print(f"z=0样本数: {len(adult_z0_idx)}")
print(f"z=1样本数: {len(adult_z1_idx)}")

# 使用狄利克雷分布生成每个客户端的数据比例
proportions = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=2)

print(f"\n狄利克雷分布参数 α = {ALPHA}")
print(f"生成的初始客户端数据比例:")
print(f"z=0的分配比例: {proportions[0]}")
print(f"z=1的分配比例: {proportions[1]}")

# 根据比例分配数据到各客户端
adult_clients_idx = []

for c in range(NUM_CLIENTS):
    client_indices = []

    # 分配z=0的数据
    n_z0 = int(proportions[0][c] * len(adult_z0_idx))
    # 确保至少有MIN_SAMPLES_PER_GROUP个z=0样本
    n_z0 = max(n_z0, MIN_SAMPLES_PER_GROUP)
    sampled_z0 = np.random.choice(adult_z0_idx, n_z0, replace=True)
    client_indices.extend(sampled_z0)

    # 分配z=1的数据
    n_z1 = int(proportions[1][c] * len(adult_z1_idx))
    # 确保至少有MIN_SAMPLES_PER_GROUP个z=1样本
    n_z1 = max(n_z1, MIN_SAMPLES_PER_GROUP)
    sampled_z1 = np.random.choice(adult_z1_idx, n_z1, replace=True)
    client_indices.extend(sampled_z1)

    # 如果总样本数仍然小于MIN_SAMPLES，补充更多数据
    while len(client_indices) < MIN_SAMPLES:
        # 随机选择补充z=0还是z=1的数据
        if np.random.random() < 0.5:
            extra = np.random.choice(adult_z0_idx, 1)
        else:
            extra = np.random.choice(adult_z1_idx, 1)
        client_indices.extend(extra)

    # 打乱客户端数据
    client_indices = np.array(client_indices)
    np.random.shuffle(client_indices)
    adult_clients_idx.append(client_indices)

# 打印客户端统计信息
print("\n客户端数据分布:")
total_assigned = 0
for i, client_idx in enumerate(adult_clients_idx):
    client_data = adult.iloc[client_idx]
    z0_count = (client_data['z'] == 0).sum()
    z1_count = (client_data['z'] == 1).sum()
    total = len(client_idx)
    total_assigned += total
    print(
        f"客户端{i + 1}: 总样本={total}, z=0: {z0_count}({z0_count / total:.1%}), z=1: {z1_count}({z1_count / total:.1%})")

print(f"\n总分配样本数: {total_assigned} (由于有放回采样，可能大于原始数据集)")

adult_num_features = len(adult.columns) - 1
adult_test = LoadData(test, 'salary', 'z')
adult_train = LoadData(adult, 'salary', 'z')
torch.manual_seed(0)
adult_info = [adult_train, adult_test, adult_clients_idx]

# COMPAS
# 0) 数据读入与基本预处理（保持不变）
sensitive_attributes   = ['sex', 'race']
categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc']
continuous_attributes  = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                    'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc',
                    'two_year_recid']
label_name = 'two_year_recid'

compas = process_csv(
    'compas', 'compas-scores-two-years.csv',
    label_name, 0,
    sensitive_attributes,           # 需要 drop 的取值：['Female', 'African-American']
    ['Female', 'African-American'],
    categorical_attributes, continuous_attributes,
    features_to_keep
)

train = compas.iloc[: int(len(compas) * .7)]
test  = compas.iloc[int(len(compas) * .7):]

# 1) Dirichlet‑α 参数（可与 Adult 区域保持一致）
NUM_CLIENTS            = 3    # 客户端数
ALPHA                  = 0.1  # Dirichlet α（越小→分布越不均匀）
MIN_SAMPLES            = 100  # 每客户端最少样本
MIN_SAMPLES_PER_GROUP  = 20   # 每客户端每个敏感组最少样本

np.random.seed(1)
torch.manual_seed(0)

# 2) 提取不同 z 组的索引
compas_z0_idx = train[train['z'] == 0].index.to_numpy()
compas_z1_idx = train[train['z'] == 1].index.to_numpy()

# 3) 生成 2×NUM_CLIENTS Dirichlet 比例矩阵
proportions = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=2)

# 4) 依比例抽样（放回），并满足最小样本数约束
compas_clients_idx = []
for c in range(NUM_CLIENTS):
    idx_c = []

    # ----- z = 0 -----
    n0 = max(int(proportions[0, c] * len(compas_z0_idx)), MIN_SAMPLES_PER_GROUP)
    idx_c.extend(np.random.choice(compas_z0_idx, n0, replace=True))

    # ----- z = 1 -----
    n1 = max(int(proportions[1, c] * len(compas_z1_idx)), MIN_SAMPLES_PER_GROUP)
    idx_c.extend(np.random.choice(compas_z1_idx, n1, replace=True))

    # ----- 补足 MIN_SAMPLES -----
    while len(idx_c) < MIN_SAMPLES:
        extra_pool = compas_z0_idx if np.random.rand() < 0.5 else compas_z1_idx
        idx_c.append(np.random.choice(extra_pool))

    idx_c = np.array(idx_c)
    np.random.shuffle(idx_c)
    compas_clients_idx.append(idx_c)

# 5) DataLoader 所需对象
compas_num_features = len(compas.columns) - 1
compas_z = len(set(compas.z))
compas_train        = LoadData(train, label_name, 'z')
compas_test         = LoadData(test,  label_name, 'z')
compas_info         = [compas_train, compas_test, compas_clients_idx]

# Bank
######################################################
### Pre-processing code (leave here for reference) ###
######################################################
# import pandas as pd
# import numpy as np
# import os
# from utils import LoadData

# df = pd.read_csv(os.path.join('bank', 'bank-full.csv'), sep = ';')
# q1 = df.age.quantile(q = 0.2)
# q1_idx = np.where(df.age <= q1)[0]
# q2 = df.age.quantile(q = 0.4)
# q2_idx = np.where((q1 < df.age) & (df.age <= q2))[0]
# q3 = df.age.quantile(q = 0.6)
# q3_idx = np.where((q2 < df.age) & (df.age <= q3))[0]
# q4 = df.age.quantile(q = 0.8)
# q4_idx = np.where((q3 < df.age) & (df.age <= q4))[0]
# q5_idx = np.where(df.age > q4)[0]
# df.loc[q1_idx, 'age'] = 0
# df.loc[q2_idx, 'age'] = 1
# df.loc[q3_idx, 'age'] = 2
# df.loc[q4_idx, 'age'] = 3
# df.loc[q5_idx, 'age'] = 4
# df.to_csv(os.path.join('bank', 'bank_cat_age.csv'))
######################################################

np.random.seed(1)
torch.manual_seed(0)
sensitive_attributes = ['age']
categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
continuous_attributes = ['balance', 'duration', 'campaign', 'pdays', 'previous']
features_to_keep = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome',
                    'balance', 'duration', 'campaign', 'pdays', 'previous', 'y']
label_name = 'y'

bank = process_csv('bank', 'bank_cat_age.csv', label_name, 'yes', sensitive_attributes, None, categorical_attributes, continuous_attributes, features_to_keep, na_values = [])
bank = bank.sample(frac=1).reset_index(drop=True)

train = bank.iloc[:int(len(bank)*.7)]
test = bank.iloc[int(len(bank)*.7):]

loan_idx = np.where(train.loan_no == 1)[0]
loan_no_idx = np.where(train.loan_no == 0)[0]
client1_idx = np.concatenate((loan_idx[:int(len(loan_idx)*.5)], loan_no_idx[:int(len(loan_no_idx)*.2)]))
client2_idx = np.concatenate((loan_idx[int(len(loan_idx)*.5):int(len(loan_idx)*.6)], loan_no_idx[int(len(loan_no_idx)*.2):int(len(loan_no_idx)*.8)]))
client3_idx = np.concatenate((loan_idx[int(len(loan_idx)*.6):], loan_no_idx[int(len(loan_no_idx)*.8):]))
np.random.shuffle(client1_idx)
np.random.shuffle(client2_idx)
np.random.shuffle(client3_idx)

bank_mean_sensitive = train['z'].mean()
bank_z = len(set(bank.z))

clients_idx = [client1_idx, client2_idx, client3_idx]

bank_num_features = len(bank.columns) - 1
bank_train = LoadData(train, label_name, 'z')
bank_test = LoadData(test, label_name, 'z')

bank_info = [bank_train, bank_test, clients_idx]