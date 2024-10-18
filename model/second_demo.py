from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_csv("heart_attack_prediction_dataset.csv")


def check_df(dataframe, head=5):
    print("############## Shape  ##################")
    print(dataframe.shape)
    print("############## Types  ##################")
    print(dataframe.dtypes)
    print("############## Head  ##################")
    print(dataframe.head(head))
    print("############## Tail  ##################")
    print(dataframe.tail(head))
    print("############## NA  ##################")
    print(dataframe.isnull().sum())
    print("############## Quantiles  ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

cat_cols = [col for col in df.columns if str(df[col].dtypes) in [
    "object", "bool"]]

# print(cat_cols)


num_but_cat = [col for col in df.columns if df[col].nunique(
) < 5 and df[col].dtypes in ["int64", "float64"]]


# print(num_but_cat)

cat_but_cardinal = [col for col in df.columns if df[col].nunique(
) > 20 and str(df[col].dtypes) in ["category", "object"]]

# print(cat_but_cardinal)

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_cardinal]

num_cols = [col for col in df.columns if col not in cat_cols]

num_cols.pop(0)

# ! kan basıncı, tansiyon nümerik değil ama ayrı ayrı işlendiği yerler de var
num_cols.remove("Blood Pressure")
print(num_cols)


# print("nümerik: "+numeric)


#! fonksiyona dökelim

# df["Heart Attack Risk"].value_counts()

# yuzdelik = 100 * df["Heart Attack Risk"].value_counts() / len(df)
# print(yuzdelik)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


# cat_summary(df, "Sex", plot=True)

#  #!!! Sayısal Değişken Analizi


# # print(df[["Age"]].describe().T)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.25, 0.50, 0.75, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)
    print("Histogram")
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    print("********************************")


# num_summary(df, "Age", plot=True)

# for col in num_cols:
#     num_summary(df, col, plot=True)

# for col in df["Blood Pressure"]:
#     print(col)

# #!!!!!!!!! Değişkenlerin Yakalanması !!!!!!!!!!!!!!!!!!!!!!!


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)
    print("Histogram")
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    print("********************************")


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# for col in num_cols:
#     num_summary(df, col, plot=True)


# df.info()

# cat_cols, num_cols, cat_but_cardinal = grab_col_names(df)
# for col in cat_cols:
#     cat_summary(df, col, plot=True)


# #!!!!!!!! Hedef Değişken Analizi !!!!!!!!!!!!!


# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 500)

# df = sns.load_dataset("titanic")

# for col in df.columns:
#     if df[col].dtypes == "bool":
#         df[col] = df[col].astype(int)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin ismini verir

    Args:
        dataframe (dataframe): değişken isimleri alınmak istenen dataframe'dir
        cat_th (int, float): nümerik fakat kategorik olan değişkenler için sınıf eşik değeri. Defaul değeri 10.
        car_th (int, float): kategorik fakat kardinal değişkenler için sınıf eşik değeri. Defaul değeri 20.

    Returns:
        cat_cols: list
            kategorik değişken listesi
        num_cols: list
            nümerik değişken listesi
        cat_but_car:list
            kategorik görünümlü kardinal değişken listesi

    Notes:
    cat_cols + num_Cols + cat_but_Car  = toplam değişken sayısı
    numeric cat_cols'un içerisindedir
    return olan 3 liste toplamı toplam değişken sayısına eşittir
    """
    cat_cols = [col for col in df.columns if str(
        df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique(
    ) < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_cardinal = [col for col in df.columns if df[col].nunique(
    ) > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_cardinal]
    numeric = [col for col in df.columns if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_cardinal)}")
    print(f"numeric: {len(numeric)}")

    return cat_cols, num_cols, cat_but_cardinal


cat_cols, num_cols, cat_but_cardinal = grab_col_names(df)
#! tansiyon için farklı bir yöntem gerekecek
cat_but_cardinal.remove("Blood Pressure")
cat_but_cardinal.remove("Patient ID")
print(cat_but_cardinal)
df["Heart Attack Risk"].value_counts()
cat_summary(df, "Heart Attack Risk")

# ##############
# ! hedef değişkenin kategorik değişkenler ile analizi
# ##########

# df.groupby("Sex")["Heart Attack Risk"].mean()


# def target_summary_with_cat(dataframe, target, categorical_col):
#     print({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}, )


# target_summary_with_cat(df, "Heart Attack Risk", "Sex")

# for col in cat_cols:
#     target_summary_with_cat(df, "Heart Attack Risk", col)


# #!!! hedef değişkenin sayısal değişkenler ile analizi
# print(df.groupby("survived")["age"].mean())
# print(df.groupby("Heart Attack Risk").agg({"Age": "mean"}))


# def target_summary_with_num(dataframe, target, numerical_col):
#     print(df.groupby(target).agg({numerical_col: "mean"}))


# target_summary_with_num(df, "Heart Attack Risk", "Age")

# for col in num_cols:
#     target_summary_with_num(df, "Heart Attack Risk", col)


# #!!!!!!!!!!!!!!!!! KORELASYON ANALİZİ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()
# print(corr)

sns.set_theme(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
# plt.show()


# #! yüksek korelasyonlu değişkenlerin silinmesi

cor_matrix = df.corr().abs()

# #! matrisi istenen şekile sokma
upper_triangle_matrix = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))


drop_list = [col for col in upper_triangle_matrix.columns if any(
    upper_triangle_matrix[col] > 0.90)]

cor_matrix[drop_list]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(
        upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


# high_correlated_cols(df)

drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

print(drop_list)
