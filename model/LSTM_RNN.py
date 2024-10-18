from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import accuracy_score, \
    f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import xgboost as xgb


df = pd.read_csv("heart_attack_prediction_dataset.csv")
df = df.rename(columns={
    "Sleep Hours Per Day": "Sleeping Hours",
    "Physical Activity Days Per Week": "Activity per Week",
    "Sedentary Hours Per Day": "Sedentary Hours",
    "Patient ID": "ID"})

continent_counts = df["Continent"].value_counts()
gender_counts = df["Sex"].value_counts()
smoking_counts = df["Smoking"].value_counts()
risk_counts = df['Heart Attack Risk'].value_counts()


del df["ID"]

df["Active Hours"] = 24 - df["Sedentary Hours"]
categorical_columns = df.select_dtypes("category").columns

bp_splitted = df["Blood Pressure"].str.split("/", expand=True)
df["systolic"] = bp_splitted[0]
df["dysystolic"] = bp_splitted[1]
df["systolic"] = df["systolic"].astype("int32")
df["dysystolic"] = df["dysystolic"].astype("int32")

del df["Blood Pressure"]

value_map = {
    "Unhealthy": 1,
    "Average": 2,
    "Healthy": 3
}
df["Diet"] = df["Diet"].map(value_map)


df["Diet"] = df["Diet"].astype("int32")

df = pd.get_dummies(
    df, columns=['Sex', 'Country', 'Continent', 'Hemisphere'], drop_first=True)
bool_columns = df.select_dtypes(include="bool").columns
for bool_col in bool_columns:
    df[bool_col] = df[bool_col].astype("int32")

X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nunique_10 = df.nunique() > 10
cols_to_scale = df.loc[:, nunique_10].columns


scaler = RobustScaler()

scaler.fit(X_train[cols_to_scale])

y.value_counts(normalize=True)


smote = SMOTE(random_state=365)
X_smote, y_smote = smote.fit_resample(X, y)

y_smote.value_counts(normalize=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2)

X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

timesteps = 100
features = 36
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(
        timesteps, features)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(X_train.shape)
print(y_train.shape)

X_train_reshaped = np.zeros((X_train.shape[0], 100, 36))

for i in range(X_train.shape[1]):  # Her bir örnek için
    original_data = X_train[i]  # Orjinal veriyi al

    # Yeniden şekillendirme
    X_train_reshaped[i, :original_data.shape[0], :original_data.shape[1]]


y_train_reshaped = np.zeros((X_train.shape[0], 100, 36))

for i in range(y_train.shape[1]):  # Her bir örnek için
    original_data = y_train[i]  # Orjinal veriyi al

    y_train_reshaped[i, :original_data.shape[0], :original_data.shape[1]]


# X_train_reshaped = X_train.values.reshape(
#     X_train.shape[0], timesteps, features)
# X_test_reshaped = X_test.values.reshape(X_test.shape[0], timesteps, features)

val_data = X_train_reshaped
val_labels = y_train_reshaped


model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print("Model çalışıyor ...")
model.fit(X_train, y_train, epochs=10, validation_data=(
    val_data, val_labels))
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy LSTM:', test_acc)
