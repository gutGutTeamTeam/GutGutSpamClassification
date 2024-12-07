import matplotlib.pyplot as plt
import pandas as pd

from config.variables import path_to_k_fold

df = pd.read_pickle(path_to_k_fold)
train_loss = (sum(df['train_loss'].tolist()))/len(df['train_loss'])
test_loss = (sum(df['test_loss'].tolist()))/len(df['test_loss'])
plt.bar(["train_loss", "test_loss"], [0.0754, 0.1024])
plt.title("average loss")
plt.show()