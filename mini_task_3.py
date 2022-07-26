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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as forestreg
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor 
from sklearn.preprocessing import MinMaxScaler

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

# увидела что width depth коррелируют и отобразила на графиках. Наверное width можно убрать
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
