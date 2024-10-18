from imblearn.over_sampling import SMOTE
import json

import numpy as np
from flask import Flask, abort, request, jsonify, render_template
from joblib import load, dump
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


dtc = load(r'C:/Users/nostr/OneDrive/Masaüstü/BitirmeProjesi\bitirme_demo/models/DecisionTreeClassifier.pkl')

df = pd.read_csv("heart_attack_prediction_dataset.csv")
df = df.rename(columns={
    "Sleep Hours Per Day": "Sleeping Hours",
    "Physical Activity Days Per Week": "Activity per Week",
    "Sedentary Hours Per Day": "Sedentary Hours",
    "Patient ID": "ID"})

del df["ID"]

df["Active Hours"] = 24 - df["Sedentary Hours"]

col = ["Age", "Cholesterol", "Income"]

numeric_columns = df.select_dtypes(include=['int16', 'float16', "int64"])

numeric_columns.drop("Heart Attack Risk", axis=1, inplace=True)

numeric_columns = df.select_dtypes(include="number")

categorical_columns = df.select_dtypes("category").columns

bp_splitted = df["Blood Pressure"].str.split("/", expand=True)
df["systolic"] = bp_splitted[0]
df["dysystolic"] = bp_splitted[1]
df["systolic"] = df["systolic"].astype("int32")
df["dysystolic"] = df["dysystolic"].astype("int32")
print(df["systolic"].dtype)
print(df["dysystolic"].dtype)

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


y.value_counts(normalize=True)


smote = SMOTE(random_state=365)
X_smote, y_smote = smote.fit_resample(X, y)

y_smote.value_counts(normalize=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2)

scaler = RobustScaler()

scaler.fit(X_train[cols_to_scale])

X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)


# pred = rf.predict(y_test)

# print(X_test.head(1))
# pred = rf.predict(X_test.values.reshape(-1, 1))
# print(pred)


avg_values = X_test.mean()


avg_data = pd.DataFrame(avg_values).transpose()
# print(avg_data)

print(avg_data["Heart Rate"])
# print(rf.predict(avg_data))
print("********************************")
print(avg_data.columns.tolist())
print("********************************")
print(avg_data['Heart Rate'])

app = Flask(__name__)


@app.route('/')
def showHomePage():
    return "This is home page"


@app.route("/rf", methods=['POST'])
def rf_api():
    # Post metoduyla gelen 'heart_rate' parametresini al ve float olarak dönüştür
    heart_rate = request.json.get('heartRate')
    if heart_rate is not None:
        heart_rate = str(heart_rate)
    else:
        abort(400, description="Missing 'Heart Rate' in JSON request")
        # heart_rate = 'Default Value'  # veya uygun bir varsayılan değer

    # Dışarıdan alınan heart_rate'i avg_data'ya ekleyelim
    avg_data['Heart Rate'] = heart_rate

    # Tahmin yapmak için avg_data'yı kullan
    # rf.predict fonksiyonuna bir liste olarak veri gönder
    prediction = rf.predict(avg_data)

    print(prediction)

    output = prediction[0]
    if output == 0:
        return "Kalp krizi riski yok"
    else:
        return "Kalp krizi riski var"


if __name__ == '__main__':
    app.run(host="0.0.0.0")


"""
['Age', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Sedentary Hours', 
'Income', 'BMI', 'Triglycerides', 'Activity per Week', 'Sleeping Hours', 'Active Hours', 'systolic', 'dysystolic', 
'Sex_Male', 'Country_Australia', 'Country_Brazil', 'Country_Canada', 'Country_China', 'Country_Colombia', 'Country_France', 
'Country_Germany', 'Country_India', 'Country_Italy', 'Country_Japan', 'Country_New Zealand', 'Country_Nigeria', 'Country_South Africa', 
'Country_South Korea', 'Country_Spain', 'Country_Thailand', 'Country_United Kingdom', 'Country_United States', 'Country_Vietnam', 
'Continent_Asia', 'Continent_Australia', 'Continent_Europe', 'Continent_North America', 'Continent_South America', 
'Hemisphere_Southern Hemisphere']
"""
