import sklearn.metrics as mt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')

data = pd.read_csv("heart_attack_prediction_dataset.csv")

df = data.copy()

print(df.head())
print(df.shape)

labels = ["Hayır = 0", "Evet = 1"]


values = df["Heart Attack Risk"].value_counts().to_numpy()


plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Sınıf Dağılımı(Kalp Krizi Riski)")
plt.show()

plt.bar(x=labels, height=values)
plt.title("Sınıf Dağılımı (Kalp Krizi Riski)")
plt.show()

print(df.isna().sum())

df[['BP_Systolic', 'BP_Diastolic']
   ] = df['Blood Pressure'].str.split('/', expand=True)

df['BP_Systolic'] = pd.to_numeric(df['BP_Systolic'])
df['BP_Diastolic'] = pd.to_numeric(df['BP_Diastolic'])

df = df.drop("Blood Pressure", axis=1)

print(df.dtypes)


encoder = LabelEncoder()


for col_name in df.columns:

    if df[col_name].dtype == "object":
        df[col_name] = encoder.fit_transform(df[[col_name]])


print(df.dtypes)
print(df.shape)


corr = df.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


X = df.drop("Heart Attack Risk", axis=1)

y = df["Heart Attack Risk"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


# Karar Ağacı

dtc_before_scaling = DecisionTreeClassifier()
dtc_before_scaling.fit(X_train, y_train)
y_predicted_train_dtc_before_scaling = dtc_before_scaling.predict(
    X_train)
dtc_train_score_before_scaling = accuracy_score(
    y_train, y_predicted_train_dtc_before_scaling)
y_predicted_train_before_scaling_dtc = dtc_before_scaling.predict(X_train)


y_predicted_test_before_scaling_dtc = dtc_before_scaling.predict(X_test)
dtc_test_score_before_scaling = accuracy_score(
    y_test, y_predicted_test_before_scaling_dtc)


dtc_after_scaling = DecisionTreeClassifier()
dtc_after_scaling.fit(X_train, y_train)
y_predicted_train_dtc = dtc_after_scaling.predict(X_train)

dtc_train_score = accuracy_score(y_train, y_predicted_train_dtc)

print("DT Train Score:", dtc_train_score)

y_predicted_test_dtc = dtc_after_scaling.predict(X_test)
dtc_test_score = accuracy_score(y_test, y_predicted_test_dtc)

print("DT Test Score:", dtc_test_score)

plt.bar(x=["dtc_train_score_before_scaling", "dtc_train_score_after_scaling"],
        height=[dtc_train_score_before_scaling, dtc_train_score])
plt.show()


plt.bar(x=["dtc test score before scaling", "dtc test score after scaling"],
        height=[dtc_test_score_before_scaling, dtc_test_score])
plt.show()


# Random Forest


rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

y_predicted_train_rfc = rfc.predict(X_train)

rfc_train_score = accuracy_score(y_train, y_predicted_train_rfc)


print(f"random forest doğruluk skoru(eğitim): {rfc_train_score} ")

y_predicted_test_rfc = rfc.predict(X_test)

rfc_test_score = accuracy_score(y_test, y_predicted_test_rfc)

print(f"random forest doğruluk skoru(test): {rfc_test_score} ")


# KNN


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_predicted_train_knn = knn.predict(X_train)

knn_train_score = accuracy_score(y_train, y_predicted_train_knn)

print(f"knn eğitim {knn_train_score}")

y_predicted_test_knn = knn.predict(X_test)

knn_test_score = accuracy_score(y_test, y_predicted_test_knn)

print(f"knn test skoru: {knn_test_score}")


# karşılaştırmalar

plt.figure(figsize=(8, 5))

labels = ["Karar Ağacı Sınıflandırıcısı",
          "Rastgele Orman Sınıflandırıcısı", "K-En Yakın Komşu "]
values = [dtc_train_score, rfc_train_score, knn_train_score]

plt.bar(x=labels, height=values)
plt.title("Tahmin Doğruluğu(Eğitim Verisi))")
plt.show()

plt.figure(figsize=(8, 5))

labels = ["Karar Ağacı Sınıflandırıcısı",
          "Rastgele Orman Sınıflandırıcısı", "K-En Yakın Komşu "]
values = [dtc_test_score, rfc_test_score, knn_test_score]

plt.bar(x=labels, height=values)
plt.title("Tahmin Doğruluğu(Test Verisi)")
plt.show()


model_train_data = [y_predicted_train_dtc,
                    y_predicted_train_rfc, y_predicted_train_knn]
model_test_data = [y_predicted_test_dtc,
                   y_predicted_test_rfc, y_predicted_test_knn]
# Train
model_train_precision_scores = []
model_train_recall_scores = []

for model_data in model_train_data:
    model_train_precision_scores.append(
        mt.precision_score(model_data, y_train))
    model_train_recall_scores.append(mt.recall_score(model_data, y_train))
labels = ["Karar Ağacı Sınıflandırıcısı",
          "Rastgele Orman Sınıflandırıcısı", "K-En Yakın Komşu "]
data = {
    'Recall': model_train_recall_scores,
    'Precision': model_train_precision_scores,

}

x = np.arange(len(labels))
width = 0.3
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, label_type="edge", padding=5)
    multiplier += 1


ax.set_title('Duyarlılık(Recall) Ve  Kesinlik(Precision) Analizi(Eğitim)')
ax.set_xticks(x + width, labels)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1.5)

plt.show()


model_test_precision_scores = []
model_test_recall_scores = []

for model_data in model_test_data:
    model_test_precision_scores.append(mt.precision_score(model_data, y_test))
    model_test_recall_scores.append(mt.recall_score(model_data, y_test))
data = {
    'Recall': model_test_recall_scores,
    'Precision': model_test_precision_scores,
}
print(data)
# {'Recall': [0.3556701030927835, 0.36619718309859156, 0.34346103038309117],
#     'Precision': [0.3678038379530917, 0.02771855010660981, 0.2771855010660981]}
labels = ["Karar Ağacı Sınıflandırıcısı",
          "Rastgele Orman Sınıflandırıcısı", "K-En Yakın Komşu "]
data = {
    'Recall': model_test_recall_scores,
    'Precision': model_test_precision_scores,
}

x = np.arange(len(labels))
width = 0.3
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, label_type="edge", padding=5)
    multiplier += 1


ax.set_title('Duyarlılık(Recall) Ve Kesinlik(Precision) Analizi(Test)')
ax.set_xticks(x + width, labels)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1.5)

plt.show()


sns.heatmap(mt.confusion_matrix(
    y_predicted_test_dtc, y_test), annot=True, fmt="d")
plt.title("Karar Ağacı Sınıflandırıcısı(Test)")
plt.show()

mt.confusion_matrix(y_predicted_test_rfc, y_test)

sns.heatmap(mt.confusion_matrix(
    y_predicted_test_rfc, y_test), annot=True, fmt="d")
plt.title("Random Forest Sınıflandırıcısı(Test)")
plt.show()

mt.confusion_matrix(y_predicted_test_knn, y_test)


sns.heatmap(mt.confusion_matrix(
    y_predicted_test_knn, y_test), annot=True, fmt="d")
plt.title("K En Yakın Komşu Sınıflandırıcısı(Test)")
plt.show()
