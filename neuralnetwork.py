import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
data = pd.read_csv(url, header=None, names=column_names)

data = data[data['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]


X = data.iloc[:, :-1].values  # Fitur
y = (data['Species'] == 'Iris-setosa').astype(int).values  # Label (0 atau 1)
