from flask import Flask, abort, request
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


file_path = "heart_attack_prediction_dataset.csv"
df = pd.read_csv(file_path)
# df.head()
df.drop_duplicates(inplace=True)
# df.shape
# df.info()

columns_to_drop = ['Hemisphere', 'Patient ID',
                   'Income', 'Country', 'Continent']
df.drop(columns_to_drop, axis=1, inplace=True)
# df.head()

df['Age'].unique()
df['Age'].isnull().sum()
# df.head()
df['Sex'].unique()
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df.rename(columns={'Sex_Male': 'is_male'}, inplace=True)
df['is_male'] = df['is_male'].astype(int)
# df.head()
df['Cholesterol'].unique()
df['Cholesterol'].isnull().sum()
df['Blood Pressure'].unique()
df['Blood Pressure'].isnull().sum()


def handle_blood_pressure_systolic(value):
    value = str(value)
    value = value.split('/')
    return int(value[0])


def handle_blood_pressure_diastolic(value):
    value = str(value)
    value = value.split('/')
    return int(value[1])


df['systolic_pressure'] = df['Blood Pressure'].apply(
    handle_blood_pressure_systolic)
df['diastolic_pressure'] = df['Blood Pressure'].apply(
    handle_blood_pressure_diastolic)
df.drop(columns='Blood Pressure', axis=1, inplace=True)
# df.head()
df['Heart Rate'].unique()
df['Heart Rate'].isnull().sum()
df['Heart Rate'].isna().sum()
# df.head()
df['Diabetes'].unique()
df['Diabetes'].isnull().sum()
df['Diabetes'].isna().sum()
# df.head()
df['Family History'].unique()
df['Family History'].isnull().sum()
df['Family History'].isna().sum()
df['Smoking'].unique()
df['Smoking'].isnull().sum()
df['Smoking'].isna().sum()
df['Obesity'].unique()
df['Obesity'].isnull().sum()
df['Obesity'].isna().sum()
df['Alcohol Consumption'].unique()
df['Alcohol Consumption'].isnull().sum()
df['Alcohol Consumption'].isna().sum()
# df.head()
df['Exercise Hours Per Week'].unique()
df['Exercise Hours Per Week'].isnull().sum()
df['Exercise Hours Per Week'].isna().sum()
df['Diet'].unique()


def handle_diet(value):
    value = str(value)

    if value == 'Unhealthy':
        return 0
    elif value == 'Average':
        return 1
    elif value == 'Healthy':
        return 2
    else:
        return np.nan


df['Diet'] = df['Diet'].apply(handle_diet)
df['Diet']
df['Diet'].unique()
# df.info()
df['Previous Heart Problems'].unique()
df['Previous Heart Problems'].isnull().sum()
df['Previous Heart Problems'].isna().sum()
df['Medication Use'].unique()
df['Medication Use'].isnull().sum()
df['Medication Use'].isna().sum()
df['Stress Level'].unique()
df['Stress Level'].isnull().sum()
df['Sedentary Hours Per Day'].unique()
df['Sedentary Hours Per Day'].isnull().sum()
df['Sedentary Hours Per Day'].isna().sum()
# df.info()
df['BMI'].unique()
df['BMI'].isnull().sum()
df['BMI'].isna().sum()
df['Triglycerides'].unique()
Triglycerides = df['Triglycerides'].value_counts(ascending=False)
Triglycerides
df['Triglycerides'].isnull().sum()
df['Triglycerides'].isna().sum()
df['Physical Activity Days Per Week'].unique()
df['Physical Activity Days Per Week'].isnull().sum()
df['Physical Activity Days Per Week'].isna().sum()
df['Sleep Hours Per Day'].unique()
df['Sleep Hours Per Day'].isnull().sum()
df['Sleep Hours Per Day'].isna().sum()

# plt.style.use('ggplot')
df['Heart Attack Risk'].value_counts().plot.pie(autopct="%1.4f%%", labels=[
    "Risk", "Not Risk"], shadow=True, textprops={'color': 'black'})
df['Heart Attack Risk'].value_counts()
# df.describe()
heart_attack_risk_sorted = df['Heart Attack Risk'].value_counts().sort_index()
# print(heart_attack_risk_sorted)
plots = []

# plots.append(sns.catplot(data=df.iloc[:, 0:5]))
# plots.append(sns.catplot(data=df.iloc[:, 5:10]))
# plots.append(sns.catplot(data=df.iloc[:, 10:15]))
# plots.append(sns.catplot(data=df.iloc[:, 15:20]))
# plots.append(sns.catplot(data=df.iloc[:, 20:22]))
# for i in range(5):
#    plots[i].set_xticklabels(rotation=90)

# HeatMap
# plt.figure(figsize = (19,10))
# Adjust layout to prevent overlapping
#    plt.show()

random_over_sampler = RandomOverSampler(sampling_strategy=1)
y = df['Heart Attack Risk']
X = df.drop(['Heart Attack Risk'], axis=1)

columns = X.columns.tolist()

X, y = random_over_sampler.fit_resample(X, y)
scaler = MinMaxScaler()
X_min_max = pd.DataFrame(scaler.fit_transform(X), columns=columns)
# print(pd.DataFrame(y).describe())

balanced_df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
balanced_df['Heart Attack Risk'].value_counts().plot.pie(autopct="%1.4f%%", labels=[
    "Risk", "Not Risk"], shadow=True, textprops={'color': 'black'})
balanced_df['Heart Attack Risk'].value_counts()
# balanced_df.describe()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
best_selection_methods = {
    'chi2': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [],  "Classifier": "RandomForestClassifier"},
    'mutual_info_classif': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": "RandomForestClassifier"},
    'lasso': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": ""},
    'PCA': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": ""},
    'RFE': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": ""},
    'Stacking': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": ""},
    'Max Voting': {"Accuracy": 0, "Precision": 0, "Recall": 0, "f1": 0, "ROC_AUC": 0, "Feature_number": 0, "Features": [], "Classifier": []},
}


def is_better_method(name, accuracy, precision, recall, f1, roc_auc, feature_num, selected_features, classifier_name):
    if accuracy > best_selection_methods[name]['Accuracy']:
        best_selection_methods[name]['Accuracy'] = accuracy
        best_selection_methods[name]['Precision'] = precision
        best_selection_methods[name]['Recall'] = recall
        best_selection_methods[name]['f1'] = f1
        best_selection_methods[name]['ROC_AUC'] = roc_auc
        best_selection_methods[name]['Feature_number'] = feature_num
        best_selection_methods[name]['Features'] = selected_features

        if name == "PCA" or name == "lasso" or name == "RFE" or name == "Stacking" or name == "Max Voting":
            best_selection_methods[name]['Classifier'] = classifier_name


# Stacking
X_train, X_test, y_train, y_test = train_test_split(
    X_min_max, y, test_size=0.2, random_state=42)
base_models = [
    ('LogisticRegression', LogisticRegression()),
    ('DecisionTreeClassifier', DecisionTreeClassifier(
        criterion='gini', max_depth=50, random_state=42, max_features=12)),
    ('RandomForestClassifier1', RandomForestClassifier(n_estimators=75, n_jobs=-1)),
    ('RandomForestClassifier2', RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ('RandomForestClassifier3', RandomForestClassifier(n_estimators=125, n_jobs=-1)),
    ('RandomForestClassifier4', RandomForestClassifier(n_estimators=150, n_jobs=-1)),
    ('GradientBoostingClassifier', GradientBoostingClassifier(
        n_estimators=125, learning_rate=0.1, max_features=15, max_depth=50)),
    ('KNeighborsClassifier', KNeighborsClassifier())
]
stacked_model = StackingClassifier(
    estimators=base_models, final_estimator=RandomForestClassifier())


stacked_model.fit(X_train, y_train)  # training the stacked_model

stacked_predictions = stacked_model.predict(
    X_test)  # making predictions with stacked_model

accuracy = accuracy_score(y_test, stacked_predictions)
precision = precision_score(y_test, stacked_predictions)
recall = recall_score(y_test, stacked_predictions)
f1 = f1_score(y_test, stacked_predictions)
roc_auc = roc_auc_score(y_test, stacked_predictions)

is_better_method("Stacking", accuracy, precision, recall, f1,
                 roc_auc, 0, [], classifier_name="RandomForestClassifier")


def print_best_selection_methods():
    for method in best_selection_methods:

        if method == "PCA" or method == "lasso" or method == "RFE" or method == "Stacking" or method == "Max Voting":
            print(
                f"Classifier: {best_selection_methods[method]['Classifier']}")

        print("Name:{}, Accuracy:{},  Precision:{}, Recall:{}, f1:{}, ROC_AUC:{}, feature_number:{},  Features:{}".format(method,
                                                                                                                          best_selection_methods[method]['Accuracy'], best_selection_methods[
                                                                                                                              method]['Precision'], best_selection_methods[method]['Recall'],
                                                                                                                          best_selection_methods[method]['f1'], best_selection_methods[method][
                                                                                                                              'ROC_AUC'], best_selection_methods[method]['Feature_number'],
                                                                                                                          best_selection_methods[
                                                                                                                              method]['Features']
                                                                                                                          ))
        print("-"*50)


#  Max Voting

# standard_scaler = StandardScaler()
# X_standard = pd.DataFrame(scaler.fit_transform(X))

# x_train, x_test, y_train, y_test = train_test_split(
#     X_standard, y, test_size=0.2, random_state=42)


# def calculate_max_voting(estimators):

#     max_voting_model = VotingClassifier(estimators=estimators)
#     max_voting_model.fit(x_train, y_train)

#     max_voting_pred = max_voting_model.predict(x_test)

#     accuracy = accuracy_score(y_test, max_voting_pred)
#     precision = precision_score(y_test, max_voting_pred)
#     recall = recall_score(y_test, max_voting_pred)
#     f1 = f1_score(y_test, max_voting_pred)
#     roc_auc = roc_auc_score(y_test, max_voting_pred)

#     is_better_method("Max Voting", accuracy, precision,
#                      recall, f1, roc_auc, 0, [], estimators)


print_best_selection_methods()
metrics = {
    'Accuracy': 0,
    'Precision': 0,
    'Recall': 0,
    'F1': 0,
    'ROC_AUC': 0
}


results = pd.DataFrame.from_dict(best_selection_methods, orient='index')
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
# print(df.head(5))
# print(best_selection_methods)


print("\n**************************  -  Columns -  **********************************\n")

print(X_train.columns.tolist())
""" 
['Age', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 'Medication Use',
'Stress Level', 'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'is_male', 'systolic_pressure', 'diastolic_pressure']
"""
print(
    "\n**************************  -  X_train  -  **********************************\n")
# dict_from_array = {item[0]: item[1] for item in X_train}

# print(dict_from_array)

# print(X_train)
first_row_dict = X_train.iloc[0].to_dict()

print(first_row_dict)

print("\n**************************  -  Print(Prediction)  -  **********************************\n")

prediciction = stacked_model.predict(X_train[:1])
print(prediciction)

print("\n**************************  -  fin -  **********************************\n")


app = Flask(__name__)


@app.route('/')
def showHomePage():
    return "This is home page"


@app.route("/rf", methods=['POST'])
def rf_api():
    # Post metoduyla gelen 'heart_rate' parametresini alma  ve float olarak dönüştürme
    heart_rate = request.json.get('heartRate')
    age = request.json.get('age')
    cholesterol = request.json.get('cholesterol')
    diabetes = request.json.get('diabetes')
    familyHistory = request.json.get('familyHistory')
    smoking = request.json.get('smoking')
    obesity = request.json.get('obesity')
    alcoholConsumption = request.json.get('alcoholConsumption')
    exerciseHoursPerWeek = request.json.get('exerciseHoursPerWeek')
    diet = request.json.get('diet')
    previousHeartProblems = request.json.get('previousHeartProblems')
    medicationUse = request.json.get('medicationUse')
    stressLevel = request.json.get('stressLevel')
    sedentaryHoursPerDay = request.json.get('sedentaryHoursPerDay')
    bmi = request.json.get('bmi')
    triglycerides = request.json.get('triglycerides')
    physicalActivityDaysPerWeek = request.json.get(
        'physicalActivityDaysPerWeek')
    sleepHoursPerDay = request.json.get('sleepHoursPerDay')
    isMale = request.json.get('isMale')
    systolicPressure = request.json.get('systolicPressure')
    diastolicPressure = request.json.get('diastolicPressure')

    if heart_rate is not None:
        heart_rate = int(heart_rate)
    else:
        abort(400, description="Missing 'Heart Rate' in JSON request")

    if age is not None:
        age = int(age)
    else:
        abort(400, description="Missing 'age' in JSON request")

    if cholesterol is not None:
        cholesterol = int(cholesterol)
    else:
        abort(400, description="Missing 'cholesterol' in JSON request")

    if diabetes is not None:
        diabetes = int(diabetes)
    else:
        abort(400, description="Missing 'diabetes' in JSON request")

    if familyHistory is not None:
        familyHistory = int(familyHistory)
    else:
        abort(400, description="Missing 'familyHistory' in JSON request")

    if smoking is not None:
        smoking = int(smoking)
    else:
        abort(400, description="Missing 'smoking' in JSON request")

    if obesity is not None:
        obesity = int(obesity)
    else:
        abort(400, description="Missing 'obesity' in JSON request")

    if alcoholConsumption is not None:
        alcoholConsumption = int(alcoholConsumption)
    else:
        abort(400, description="Missing 'alcoholConsumption' in JSON request")

    if exerciseHoursPerWeek is not None:
        exerciseHoursPerWeek = float(exerciseHoursPerWeek)
    else:
        abort(400, description="Missing 'exerciseHoursPerWeek' in JSON request")

    if diet is not None:
        diet = str(diet)
    else:
        abort(400, description="Missing 'diet' in JSON request")

    if previousHeartProblems is not None:
        previousHeartProblems = int(previousHeartProblems)
    else:
        abort(400, description="Missing 'previousHeartProblems' in JSON request")

    if medicationUse is not None:
        medicationUse = int(medicationUse)
    else:
        abort(400, description="Missing 'medicationUse' in JSON request")

    if stressLevel is not None:
        stressLevel = int(stressLevel)
    else:
        abort(400, description="Missing 'stressLevel' in JSON request")

    if sedentaryHoursPerDay is not None:
        sedentaryHoursPerDay = float(sedentaryHoursPerDay)
    else:
        abort(400, description="Missing 'sedentaryHoursPerDay' in JSON request")

    if bmi is not None:
        bmi = float(bmi)
    else:
        abort(400, description="Missing 'bmi' in JSON request")

    if triglycerides is not None:
        triglycerides = int(triglycerides)
    else:
        abort(400, description="Missing 'triglycerides' in JSON request")

    if physicalActivityDaysPerWeek is not None:
        physicalActivityDaysPerWeek = int(physicalActivityDaysPerWeek)
    else:
        abort(400, description="Missing 'physicalActivityDaysPerWeek' in JSON request")

    if sleepHoursPerDay is not None:
        sleepHoursPerDay = float(sleepHoursPerDay)
    else:
        abort(400, description="Missing 'sleepHoursPerDay' in JSON request")

    if isMale is not None:
        isMale = int(isMale)
    else:
        abort(400, description="Missing 'isMale' in JSON request")

    if systolicPressure is not None:
        systolicPressure = int(systolicPressure)
    else:
        abort(400, description="Missing 'systolicPressure' in JSON request")

    if diastolicPressure is not None:
        diastolicPressure = int(diastolicPressure)
    else:
        abort(400, description="Missing 'diastolicPressure' in JSON request")

    X_train[:1]['Heart Rate'] = heart_rate
    X_train[:1]['Age'] = age
    X_train[:1]['Cholesterol'] = cholesterol
    X_train[:1]['Cholesterol'] = cholesterol
    X_train[:1]['Diabetes'] = diabetes
    X_train[:1]['Family History'] = familyHistory
    X_train[:1]['Smoking'] = smoking
    X_train[:1]['Obesity'] = obesity
    X_train[:1]['Alcohol Consumption'] = alcoholConsumption
    X_train[:1]['Exercise Hours Per Week'] = exerciseHoursPerWeek
    X_train[:1]['Diet'] = diet
    X_train[:1]['Previous Heart Problems'] = previousHeartProblems
    X_train[:1]['Medication Use'] = medicationUse
    X_train[:1]['Stress Level'] = stressLevel
    X_train[:1]['Sedentary Hours Per Day '] = sedentaryHoursPerDay
    X_train[:1]['BMI'] = bmi
    X_train[:1]['Triglycerides '] = triglycerides
    X_train[:1]['Triglycerides '] = triglycerides
    X_train[:1]['Physical Activity Days Per Week'] = physicalActivityDaysPerWeek
    X_train[:1]['Sleep Hours Per Day '] = sleepHoursPerDay
    X_train[:1]['is_male'] = isMale
    X_train[:1][' systolic_pressure'] = systolicPressure
    X_train[:1][' diastolic_pressure'] = diastolicPressure

    first_row = X_train[:1]

    # MinMaxScaler oluşturma
    scaler = MinMaxScaler()

    first_row_scaled = scaler.fit_transform(first_row)
    prediction = stacked_model.predict(first_row_scaled)

    print(prediction)

    output = prediction[0]
    if output == 0:
        return "Heart Attack Risk Detected"
    else:
        return "No Risk of Heart Attack"


if __name__ == '__main__':
    app.run(host="0.0.0.0")
