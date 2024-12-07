import numpy as np
import optuna.visualization as vis
import optuna
import pandas as pd

import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config.variables import *
from train.dataset import TextDataset
from train.rnnModel import RNNClassifier
from train.trainer import RNNPlus

'''
***********************dealing data**********************************
'''
# read source data
df = pd.read_pickle(path_to_source)

X = df[text_name].to_numpy() # Feature vectors,384 dimensions
y = df[tag_name].to_numpy() # Labels (e.g., spam/ham)
X = np.vstack(X)
X = X.reshape(len(y), max_sentences * dim)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X = X.reshape(len(y), max_sentences, dim)

# 分训练集（后续还要smote 才能使用）
#returns
(X_train, X_test,
 y_train, y_test) \
    = (train_test_split(X, y, test_size=0.3, random_state=42))
# \qwq it is maybe a wrong with the shape of y

X_train = X_train

X_test = X_test


del df
del X
del y


def objective(trial):
    # 超参数搜索范围
    # hidden_size_for_fc = trial.suggest_int("hidden_size_for_fc", 8, 32, step=8)
    # num_layers = trial.suggest_int("num_layers", 2, 3)
    num_layers = 2

    hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16) # 后面有依赖

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-1)
    num_epochs = trial.suggest_int("num_epochs", 40, 70)
    dropout = trial.suggest_float("dropout", 0.0,0.5, step=0.1)


    '''
    ************************* init pytorch dataset ******************************
    '''
    # Initialize dataset and dataloader
    dataset = TextDataset(X_train, y_train, max_sentences, dim, is_test=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, max_sentences, dim, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    '''
    *************************Model initialization******************************
    '''

    model = RNNClassifier(input_size, hidden_size, num_layers, batch_size, num_epochs, num_classes, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_pp = RNNPlus(model,criterion,optimizer,device,dataloader,test_dataloader)
    train_loss_array,test_loss_array = model_pp.epochs_t_e(num_epochs)

    # p = 0.3
    # return p * train_loss_array[-1]+ (1 - p)*test_loss_array[-1]
    return train_loss_array[-1]


study = optuna.create_study(direction="minimize", study_name="hello")
study.optimize(objective, n_trials=70)

# 输出最佳超参数
print("Best trial:")
print(study.best_trial.params)
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()

import json
# 保存最佳超参数到 JSON 文件
best_trial_params = study.best_trial.params
with open(path_to_best_parameter, 'w') as f:
    f.write(json.dumps(best_trial_params, indent=4))



