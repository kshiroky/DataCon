"""
task:
Формулу на дескрипторы
Параметры
Pymatgen
Какой вклад что вносит
Нормализовать
Разброс сплит одинаковый
Кросс валидация
На основе формулы определял два числа
Km, Kcat
Задание (последовательность указанных действий определяется вами, обратите внимание на важность этого момента):
• Разработать дескрипторы материалов на основании имеющихся в базе данных (pymatgen и другие в помощь), feature importance 
• Нормализовать данные и подготовить для построения алгоритма машинного обучения
• Выбрать лучший алгоритм МО на основании метрик точности и ошибок и !5-fold cross-validation
! Спойлер! Алгоритм из статьи не лучший
• Сделать оптимизацию гиперпараметров моделей МО
• Показать интерпретируемость модели
Предсказываемые свойства: Km, Kcat
Результат:
Рабочий код, выполняющий все эти функции и выводящий читаемые и понятные графики.
Данные:
Это данные собранные из статей по нанозимам с пероксидазной активностью. Они включают в себя данные о материале (состав, размеры, структура, условия синтеза и анализа) и его каталитической активности по кинетике Михаэлиса. 
Обозначения параметров:
formula - химическая формула
Km - константа Михаэлис-Ментен, mM
Kcat - константа каталитической активности, равная Vmax/Ccat, s-1
Syngony - кристаллическая система образца, категориальный признак (0-аморф, 1-7 системы)
length - длина наноматериала, нм
width - ширина наноматериала, нм
depth высота наноматериала, нм
pol - категориальный признак, показывающий наличие полимера в синтезе (0-нет, 1-да)
surf - категориальный признак, показывающий наличие нейтральных ПАВ в синтезе (0-нет, 1-да)
Subtype - субстрат, на котором проводили измерение каталитической активности 
ph - ph буфера, в котором измеряли каталитическую активность
temp - температура, при которой измеряли каталитическую активность
Cper - концентрация H2O2 (mM), при которой измеряли каталитическую активность
Csub - концентрация хромогенного субстрата (mM), при которой измеряли каталитическую активность
Ccat - концентрация наночастиц (mkg/ml), при которой измеряли каталитическую активность
"""



pip install category-encoders
pip install shap


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor as forestreg
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler 
from catboost import CatBoostClassifier, Pool, cv
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import TargetEncoder
import shap

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split as splt

from sklearn.model_selection import cross_val_predict


url = 'https://raw.githubusercontent.com/kshiroky/DataCon/main/task%203.csv'

raw_data = pd.read_csv(url, delimiter = ',')
column_ls = raw_data.columns

# удалила пробелы в Kcat и перенеса во float
raw_data['Kcat'] = raw_data.Kcat.str.replace(' ','').astype(float)
# raw_data.dtypes
# raw_data.head(10)

# корреляционная матрица
cols_to_analyse= list(set(column_ls) - set(['formula', 'Subtype']))
data_corr=raw_data[cols_to_analyse].corr()

plt.figure(figsize = (10,6))
sns.heatmap(data_corr)
plt.show()

# увидела что width depth коррелируют и отобразила на графиках. Наверное depth можно убрать
coreletion_w_and_d = list(set(['width', 'depth']))
sns.pairplot(raw_data[coreletion_w_and_d], size=3)
plt.show()

# отобразила статистику по колонкам
print(raw_data.describe())

# убрать ненужные стобцы
exemplary_df = raw_data.drop(['depth'], axis=1)

# замена категориальных свойств на численные
exemplary_df['formula'] = exemplary_df['formula'].astype('category').cat.codes
exemplary_df['Subtype'] = exemplary_df['Subtype'].astype('category').cat.codes

# нашли выброс
sns.boxplot(data=exemplary_df, orient="h")
plt.show()



# отстойная усатая коробка без выброса
exemplary_df.drop(exemplary_df[exemplary_df['Kcat'] >= 70000.0].index, inplace = True)
sns.boxplot(data=exemplary_df, orient="h")
plt.show()


# нормализация MinMax
x = exemplary_df.values
cols = exemplary_df.columns
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_df = pd.DataFrame(x_scaled, columns=cols)
print(norm_df)

# репрезентация нормировки по желанию: 
# for i in cols:
#     print(i)
#     plt.figure()
#     plt.hist(df[i])
#     plt.title(i)
#     plt.show()


#split data before model selection
km_X_train, km_X_test, km_y_train, km_y_test = splt(norm_df.drop(['Km','Kcat'], axis = 1), norm_df['Km'], shuffle = False, random_state = 50, test_size = 0.3)
kcat_X_train, kcat_X_test, kcat_y_train, kcat_y_test = splt(norm_df.drop(['Km','Kcat'], axis = 1), norm_df['Kcat'], shuffle = False, random_state = 50, test_size = 0.3)
x_data = norm_df.drop(['Km','Kcat'], axis = 1)
km_y_data = norm_df['Km']
kcat_y_data = norm_df['Kcat']

#evaluate the shape of splitted data
print(km_X_train.shape, km_y_train.shape)
print(km_X_test.shape, km_y_test.shape)
print(kcat_X_train.shape, kcat_y_train.shape)
print(kcat_X_test.shape, kcat_y_test.shape)
print(x_data.shape, km_y_data.shape)
print(x_data.shape, kcat_y_data.shape)

#Km model parameters
km_forest = forestreg(n_estimators = 100, random_state = 50, oob_score = True)
km_forest.fit(km_X_train, km_y_train)
#visualize the results of training
km_plot = plt.figure(figsize = (10,6))
sns.scatterplot(np.log10(km_y_train), np.log10(km_forest.oob_prediction_))
plt.xlabel('real')
plt.ylabel('predict')
plt.title('prediciton without cross-validation for Km')
x_lin = np.linspace(-7, 2, 100)
y_lin = x_lin
plt.plot(x_lin, y_lin, 'r')
plt.show()


#cross-validation
k_fold_forest = KFold(n_splits = 5)
km_cv_pred = cross_val_predict(km_forest, x_data, km_y_data, cv = k_fold_forest)
km_cv_for = cross_val_score(km_forest, x_data, km_y_data, cv = k_fold_forest) #change X_train, y_train if needed
km_cv_plot = plt.figure(figsize = (10,6))
sns.scatterplot(km_y_data, km_cv_pred)
plt.xlabel('real')
plt.ylabel('predict')
plt.title('cross-validation prediction for Km')
plt.show()

score_plot = plt.figure(figsize = (10,6)) 
sns.distplot(km_cv_for, bins = 5)
plt.title('forest scores distribution for Km')
plt.show()


#Kcat model parameters
kcat_forest = forestreg(n_estimators = 100, random_state = 50, oob_score = True)
kcat_forest.fit(kcat_X_train, kcat_y_train)

#visualize the results of training
kcat_for_plt = plt.figure(figsize = (10,6))
sns.scatterplot(np.log10(kcat_y_train), np.log10(kcat_forest.oob_prediction_))
plt.title('no cross-val random forest for Kcat')
plt.show()

#cross-validation
k_fold_forest = KFold(n_splits = 5)
kcat_cv_for = cross_val_score(kcat_forest, x_data, kcat_y_data, cv = k_fold_forest) #change X_train, y_train if needed
kcat_cv_pred = cross_val_predict(kcat_forest, x_data, kcat_y_data, cv = k_fold_forest)

kcat_cv_plot = plt.figure(figsize = (10,6)) 
sns.scatterplot(kcat_y_data, kcat_cv_pred)
plt.xlabel('real')
plt.ylabel('predict')
plt.title('cross-validation prediction for Kcat')
plt.show()

score_plot = plt.figure(figsize = (10,6)) 
sns.distplot(kcat_cv_for, bins = 5)
plt.title('forest scores distribution for Kcat')
plt.show()

# CatBoostRegressor leanring

# MODEL FOR KM WITH CATBOOST

# CatBoostRegressor leanring
cbr = CatBoostRegressor(iterations=100, learning_rate=1, depth=2)
cbr.fit(km_X_train, km_y_train, plot=True)
km_train = km_X_train.merge(km_y_train, right_index = True, left_index = True)
categorical_features_indices = np.where(km_train.dtypes != np.float)[0]

#feature importance according to cat
shap.initjs()
explainer = shap.TreeExplainer(cbr)
shap_values = explainer.shap_values(Pool(km_X_train, km_y_train, cat_features=categorical_features_indices))
shap.force_plot(explainer.expected_value, shap_values[0,:], km_X_train.iloc[0,:], matplotlib=True, show=True)
shap.summary_plot(shap_values, km_X_train, plot_type="bar")

# prediction according to cat
Km_y_pred_cbr = cbr.predict(km_X_test)

plt.figure(figsize=(10,10))
sns.regplot(km_y_test, Km_y_pred_cbr, fit_reg=True, scatter_kws={"s": 100})

# getting RMSE and bias
print(cbr.get_best_score())
print(cbr.get_scale_and_bias())


# MODEL FOR KCAT WITH CATBOOST
# kcat_X_train, kcat_X_test, kcat_y_train, kcat_y_test = splt(norm_df.drop(['Km','Kcat'], axis = 1), norm_df['Kcat'], shuffle = False, random_state = 50, test_size = 0.3)
# CatBoostRegressor leanring
cbr.fit(kcat_X_train, kcat_y_train, plot=True)
kcat_train = kcat_X_train.merge(kcat_y_train, right_index = True, left_index = True)
categorical_features_indices = np.where(kcat_train.dtypes != np.float)[0]

#feature importance according to cat
shap.initjs()
explainer = shap.TreeExplainer(cbr)
shap_values = explainer.shap_values(Pool(kcat_X_train, kcat_y_train, cat_features=categorical_features_indices))
shap.force_plot(explainer.expected_value, shap_values[0,:], kcat_X_train.iloc[0,:], matplotlib=True, show=True)
shap.summary_plot(shap_values, kcat_X_train, plot_type="bar")

# prediction according to cat
Kcat_y_pred_cbr = cbr.predict(km_X_test)

plt.figure(figsize=(10,10))
sns.regplot(kcat_y_test, Kcat_y_pred_cbr, fit_reg=True, scatter_kws={"s": 100})

# getting RMSE and bias
print(cbr.get_best_score())
print(cbr.get_scale_and_bias())