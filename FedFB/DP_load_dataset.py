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
adult_test_df = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
adult_test_df['native-country_ Holand-Netherlands'] = 0
adult_test_df = adult_test_df[adult.columns]

ADULT_NUM_CLIENTS = 5

def _dirichlet_split(z0_idx, z1_idx, num_clients, alpha, min_samples, min_samples_per_group):
    """Split indices using a Dirichlet distribution."""
    proportions = np.random.dirichlet([alpha] * num_clients, size=2)
    clients_idx = []
    for c in range(num_clients):
        idx_c = []
        n0 = max(int(proportions[0, c] * len(z0_idx)), min_samples_per_group)
        idx_c.extend(np.random.choice(z0_idx, n0, replace=True))
        n1 = max(int(proportions[1, c] * len(z1_idx)), min_samples_per_group)
        idx_c.extend(np.random.choice(z1_idx, n1, replace=True))
        while len(idx_c) < min_samples:
            extra_pool = z0_idx if np.random.rand() < 0.5 else z1_idx
            idx_c.append(np.random.choice(extra_pool))
        idx_c = np.array(idx_c)
        np.random.shuffle(idx_c)
        clients_idx.append(idx_c)
    return clients_idx


def make_adult_info(alpha=0.1, seed=1):
    """Generate Adult dataset split with Dirichlet parameter ``alpha``."""
    np.random.seed(seed)
    adult_z0_idx = adult[adult['z'] == 0].index.to_numpy()
    adult_z1_idx = adult[adult['z'] == 1].index.to_numpy()
    clients_idx = _dirichlet_split(
        adult_z0_idx,
        adult_z1_idx,
        ADULT_NUM_CLIENTS,
        alpha,
        100,
        20,
    )
    train_ds = LoadData(adult, 'salary', 'z')
    test_ds = LoadData(adult_test_df, 'salary', 'z')
    num_features = len(adult.columns) - 1
    return [train_ds, test_ds, clients_idx], num_features

# default split
adult_info, adult_num_features = make_adult_info()

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
    sensitive_attributes,
    ['Female', 'African-American'],
    categorical_attributes, continuous_attributes,
    features_to_keep
)

_compas_train_df = compas.iloc[: int(len(compas) * .7)]
_compas_test_df  = compas.iloc[int(len(compas) * .7):]

COMPAS_NUM_CLIENTS = 3

def make_compas_info(alpha=0.1, seed=1):
    """Generate COMPAS dataset split with Dirichlet parameter ``alpha``."""
    np.random.seed(seed)
    compas_z0_idx = _compas_train_df[_compas_train_df['z'] == 0].index.to_numpy()
    compas_z1_idx = _compas_train_df[_compas_train_df['z'] == 1].index.to_numpy()
    clients_idx = _dirichlet_split(
        compas_z0_idx,
        compas_z1_idx,
        COMPAS_NUM_CLIENTS,
        alpha,
        100,
        20,
    )
    train_ds = LoadData(_compas_train_df, label_name, 'z')
    test_ds = LoadData(_compas_test_df, label_name, 'z')
    num_features = len(compas.columns) - 1
    num_groups = len(set(compas.z))
    return num_groups, num_features, [train_ds, test_ds, clients_idx]

# default split
compas_z, compas_num_features, compas_info = make_compas_info()

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

bank_train_df = bank.iloc[:int(len(bank)*.7)]
bank_test_df = bank.iloc[int(len(bank)*.7):]

loan_idx = np.where(bank_train_df.loan_no == 1)[0]
loan_no_idx = np.where(bank_train_df.loan_no == 0)[0]
client1_idx = np.concatenate((loan_idx[:int(len(loan_idx)*.5)], loan_no_idx[:int(len(loan_no_idx)*.2)]))
client2_idx = np.concatenate((loan_idx[int(len(loan_idx)*.5):int(len(loan_idx)*.6)], loan_no_idx[int(len(loan_no_idx)*.2):int(len(loan_no_idx)*.8)]))
client3_idx = np.concatenate((loan_idx[int(len(loan_idx)*.6):], loan_no_idx[int(len(loan_no_idx)*.8):]))
np.random.shuffle(client1_idx)
np.random.shuffle(client2_idx)
np.random.shuffle(client3_idx)

bank_mean_sensitive = bank_train_df['z'].mean()
bank_z = len(set(bank.z))

clients_idx = [client1_idx, client2_idx, client3_idx]

bank_num_features = len(bank.columns) - 1
bank_train = LoadData(bank_train_df, label_name, 'z')
bank_test = LoadData(bank_test_df, label_name, 'z')

bank_info = [bank_train, bank_test, clients_idx]
