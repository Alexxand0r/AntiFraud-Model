import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Загрузка данных из CSV файла
df_data = pd.read_csv("url_data.csv")
df_data['target'] = pd.get_dummies(df_data['status'])["legitimate"].astype('int')
df_data.drop('status',axis = 1, inplace=True)
#Отбор колонок, которые почти не влияют на результат
likely_cat = {}
for var in df_data.iloc[:,1:].columns:
    likely_cat[var] = 1.*df_data[var].nunique()/df_data[var].count() < 0.002
num_cols = []
cat_cols = []
for col in likely_cat.keys():
    if (col!='target'):
        num_cols.append(col)
    else:
        cat_cols.append(col)
# Построение столбчатой диаграммы распределения средних значений
df_distr =df_data.groupby('target')[num_cols].mean().reset_index().T
df_distr.rename(columns={0:"legitimate",1: "phishing"}, inplace=True)
plt.rcParams['axes.facecolor']='w'
ax = df_distr[1:-3][["legitimate","phishing"]].plot(kind='bar', title ="Распределение средних значений", figsize=(12, 8), legend=True, fontsize=12)
ax.set_xlabel("Цифровые параметры", fontsize=14)
ax.set_ylabel("Средние значения", fontsize=14)
plt.show()
# Разделение данных на обучающую и тестовую выборки
X= df_data.iloc[: , 1:-1]
y= df_data['target']
train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)
# Масштабирование признаков
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(train_x.values)
x_test =  scaler.fit_transform(test_x.values)
# Преобразование признаков в тензор
x_tensor =  torch.from_numpy(x_train).float()
y_tensor =  torch.from_numpy(train_y.values.ravel()).float()
xtest_tensor =  torch.from_numpy(x_test).float()
ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float()
#Создание датасетов и загрузчиков данных
bs = 64
y_tensor = y_tensor.unsqueeze(1)
train_ds = TensorDataset(x_tensor, y_tensor)
train_dl = DataLoader(train_ds, batch_size=bs)
ytest_tensor = ytest_tensor.unsqueeze(1)
test_ds = TensorDataset(xtest_tensor, ytest_tensor)
test_loader = DataLoader(test_ds, batch_size=bs)
#Задание параметров модели
n_input_dim = train_x.shape[1]
n_hidden1 = 400
n_hidden2 = 200
n_output =  1
# Определение архитектуры модели
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1)
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output)
        self.conv1 = nn.Conv3d(n_input_dim, n_output, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x
# Создание экземпляра модели
model = ChurnModel()
#Определение некоторых гиперпараметров модели
loss_func = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 200
#Обучение модели
model.train()
train_loss = []
for epoch in range(epochs):
    for xb, yb in train_dl:
        y_pred = model(xb)
        loss = loss_func(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
# Построение графика значений потерь по эпохам
plt.plot(train_loss)
plt.show()
#Тестирование модели
y_pred_list = []
model.eval()
with torch.no_grad():
    for xb_test, yb_test in test_loader:
        y_test_pred = model(xb_test)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.detach().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
ytest_pred = list(itertools.chain.from_iterable(y_pred_list))
#Вывод информации об обучении и тестировании
y_true_test = test_y.values.ravel()
print("Precision модели:\t"+str(precision_score(y_true_test,ytest_pred)))
print("Recall модели:\t"+str(recall_score(y_true_test,ytest_pred)))
print("F1 Score модели:\t"+str(f1_score(y_true_test,ytest_pred)))
# Сохранение весов модели в файл
torch.save(model.state_dict(), 'model.pth')
