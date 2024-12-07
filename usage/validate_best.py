import numpy as np
import pandas as pd
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from config.variables import *
from train.conv_lstm import RNNClassifier
from train.dataset import TextDataset
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


with open(path_to_best_parameter) as f:
    hp_params = json.load(f)

batch_size = hp_params['batch_size']
hidden_size = hp_params['hidden_size']
num_epochs = hp_params['num_epochs']
num_layers = 2
learning_rate = hp_params['learning_rate']
dropout = hp_params['dropout']
"""
***************** 进行 K-fold 交叉验证******************************
"""
kf = KFold(n_splits=5, shuffle=True, random_state=42)

k_test_losses = []
k_train_losses = []
count_runs = 0
# 进行 K-fold 交叉验证
for train_idx, test_idx in kf.split(X):
    # 获取训练和验证数据
    """
    ************************* dealing data ****************************
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    """
    ******************* pytorch dataset *************************
    """
    dataset = TextDataset(X_train, y_train, max_sentences, dim, is_test=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, max_sentences, dim, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    """early_stop_flag
    ***********************创建模型*********************
    """
    model = RNNClassifier(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, num_classes=num_classes, batch_size=batch_size, dropout=dropout,
                          epoch=num_epochs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_pp = RNNPlus(model, criterion, optimizer, device, dataloader, test_dataloader, stop_delta=0.01,
                       stop_patient=7)

    """
    ****************创建模型*********************
    """
    train_loss_array, test_loss_array = model_pp.epochs_t_e(num_epochs,count_runs)
    count_runs += 1
    k_test_losses.append(sum(test_loss_array) / len(test_loss_array))
    k_train_losses.append(sum(train_loss_array) / len(train_loss_array))

fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), layout='constrained')
axs[0].set_title("total train loss")
axs[1].set_title("log total train loss")
axs[0].plot(k_test_losses, label="train_loss")
axs[0].plot(k_train_losses, label="test_loss")
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')
axs[0].legend()

axs[1].plot(k_test_losses, label="train_loss")
axs[1].plot(k_train_losses, label="test_loss")
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('loss')
axs[1].set_yscale('log')
axs[1].legend()
plt.show()

_, ls = plt.subplots(1, 1, figsize=(5, 2.7), layout='constrained')
train_loss = (sum(k_train_losses)) / len(k_train_losses)
test_loss = (sum(k_test_losses)) / len(k_test_losses)
ls.bar(["train_loss", "test_loss"], [train_loss, test_loss])
ls.set_title("avg_loss")
ls.set_ylabel('loss')
ls.show()
"""
df = pd.DataFrame({'train_loss': train_losses, 'test_loss': test_losses})
df.to_pickle(path_to_k_fold)
"""
