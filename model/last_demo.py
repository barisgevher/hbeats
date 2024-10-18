import pickle
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

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


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

col = ["Age", "Cholesterol", "Income"]

# toplam yaş kolestrol ve geliri hesaplama
for column in col:
    n = len(df)

    column_sum = df[column].sum()

    mean = column_sum / n

    print(f"Mean of {column}: {mean}")

#! standart sapma

col = ["Age", "Cholesterol", "Income"]

for i in col:
    n = len(df[i])

    x = df[i]
    x_mean = np.mean(df[i])

    subtraction = x - x_mean

    squared_differences = subtraction ** 2

    sum_of_squares = np.sum(squared_differences)

    division = sum_of_squares / n-1

    standard_deviation = np.sqrt(division)

    print(f"Standard deviation of {i}: {standard_deviation}")


#! görselleştir

plt.figure(figsize=(5, 4))

count_plot_columns = ["Sex", "Diet", "Country", "Continent", "Hemisphere"]
for i in count_plot_columns:
    sns.countplot(x=i, data=df)
    plt.title(f'This is a countplot of {i}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()


pie_plot_columns = ["Heart Attack Risk", "Smoking",
                    "Diabetes", "Previous Heart Problems"]

for pie_columns in pie_plot_columns:
    plt.figure(figsize=(5, 4))
    counts = df[pie_columns].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title(f'Pie Plot of {pie_columns}')
    # plt.show()


#! korelasyon matrisi
numeric_columns = df.select_dtypes(include=['int16', 'float16', "int64"])

numeric_columns.drop("Heart Attack Risk", axis=1, inplace=True)

correlation = numeric_columns.corrwith(df['Heart Attack Risk'])
score = correlation


plt.figure(figsize=(10, 6))
plt.barh(score.index, score)
plt.xlabel('Correlation with Heart Attack Risk')
plt.ylabel('Columns')
plt.title('Correlation of Numeric Columns with Heart Attack Risk')
# plt.show()


histogram_columns = ["Income", "BMI", "Age"]

for hist_columns in df.columns:
    plt.figure(figsize=(4, 3))
    plt.hist(df[hist_columns], color='yellow', edgecolor='red')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()


plt.figure(figsize=(10, 6))

sns.pointplot(data=df, x='Continent', y='Income')

plt.xlabel('Continent')
plt.ylabel('Income')
plt.title('Income Variation Across Continents')
# plt.show()


x_column = 'Continent'

y_column = 'Heart Attack Risk'

sns.catplot(data=df, x=x_column, y="Income", kind="point",
            hue="Sex", join=False, dodge=True)

plt.xticks(rotation=45)

# plt.show()

numeric_columns = df.select_dtypes(include="number")

correlation_score = numeric_columns.corr()

plt.figure(figsize=(25, 10))
sns.heatmap(correlation_score, annot=True)
plt.xticks(rotation=30)
# plt.show()

categorical_columns = df.select_dtypes("category").columns

bp_splitted = df["Blood Pressure"].str.split("/", expand=True)
df["systolic"] = bp_splitted[0]
df["dysystolic"] = bp_splitted[1]
df["systolic"] = df["systolic"].astype("int32")
df["dysystolic"] = df["dysystolic"].astype("int32")
print(df["systolic"].dtype)
print(df["dysystolic"].dtype)

del df["Blood Pressure"]


print(df["Diet"].unique())

value_map = {
    "Unhealthy": 1,
    "Average": 2,
    "Healthy": 3
}


df["Diet"] = df["Diet"].map(value_map)

print(df["Diet"].unique())


df["Diet"] = df["Diet"].astype("int32")

df = pd.get_dummies(
    df, columns=['Sex', 'Country', 'Continent', 'Hemisphere'], drop_first=True)
bool_columns = df.select_dtypes(include="bool").columns
for bool_col in bool_columns:
    df[bool_col] = df[bool_col].astype("int32")


X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nunique_10 = df.nunique() > 10
cols_to_scale = df.loc[:, nunique_10].columns

y.value_counts(normalize=True)


smote = SMOTE(random_state=365)
X_smote, y_smote = smote.fit_resample(X, y)

y_smote.value_counts(normalize=True)

plt.pie(y_smote.value_counts(), autopct="%1.1f%%", startangle=140)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2)
print(f"""

Rows in X_train: {X_train.shape[0]}
Rows in y_train: {y_train.shape[0]}

Rows in X_test: {X_test.shape[0]}
Rows in y_test: {y_test.shape[0]} 

""")


scaler = RobustScaler()

scaler.fit(X_train[cols_to_scale])

X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


clf = GaussianNB()

clf.fit(X_train, y_train)

clf_score = clf.score(X_test, y_test)

print(clf_score * 100)


lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_score = lr.score(X_test, y_test)

print(lr_score * 100)

dtr = DecisionTreeClassifier()

dtr.fit(X_train, y_train)

dtr_score = dtr.score(X_test, y_test)

print(dtr_score * 100)

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)

print(rf_score * 100)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_score = knn.score(X_test, y_test)

print(knn_score * 100)


svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
svm_score = svm_model.score(X_test, y_test)

xgb_model = xgb.XGBClassifier()
num_folds = 5


# Cross-validation ile model performansını değerlendirme
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=num_folds)

# XGBoost modelinin performansını yazdırma
print("XGBoost:")
print(f"Mean Accuracy: {np.mean(xgb_scores)}")
print(f"Standard Deviation of Accuracy: {np.std(xgb_scores)}")


# Diğer kısımları buraya almadım, sadece yapay sinir ağı ile ilgili kısmı ekliyorum

# Yapay sinir ağı modeli oluşturma
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Modeli eğitme
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Model performansını değerlendirme
_, nn_accuracy = nn_model.evaluate(X_test, y_test)

# Yapay sinir ağı modelinin performansını yazdırma
print("Neural Network:")
print(f"Accuracy: {nn_accuracy}")

models = [clf, lr, dtr, rf, knn, svm_model]

model_predictions = {}

for model in models:
    model_name = str(model).split("(")[0]

    model_predictions[model_name] = model.predict(X_test)

print(model_predictions)


model_evaluation = {}


for model, preds in model_predictions.items():
    model_evaluation[model] = [
        round(accuracy_score(y_test, preds) * 100, 2),
        round(f1_score(y_test, preds) * 100, 2),
        round(precision_score(y_test, preds) * 100, 2),
        round(recall_score(y_test, preds) * 100, 2),
    ]


models = [
    GaussianNB(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    svm.SVC()
]


model_scores = {}
for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, X_train, y_train, cv=num_folds)
    model_scores[model_name] = scores


for model_name, scores in model_scores.items():
    print(f"{model_name}:")
    print(f"Mean Accuracy: {np.mean(scores)}")
    print(f"Standard Deviation of Accuracy: {np.std(scores)}")


models = {
    "clf": clf,
    "lr": lr,
    "dtr": dtr,
    "rf": rf,
    "knn": knn,
    "svm_model": svm_model,
    "xgb_model": xgb_model,
    "nn_model": nn_model
}

for model_name, model in models.items():

    model_class_name = model.__class__.__name__

    filename = f"{model_class_name}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(model, f)
