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

url = 'https://raw.githubusercontent.com/kshiroky/DataCon/main/task%203.csv'

raw_data = pd.read_csv(url, delimiter = ',')
column_ls = raw_data.columns

#turn categorial features into numbers
def indexer(column, name_of_col = 'feature'):
    try: 
        for i in column.tolist(): float(i)
        return column
    except:
        if type(column) == type(pd.Series()):
            col_dc = {}
            uniq_ls = column.unique()
            for k in range(len(uniq_ls)): col_dc[uniq_ls[k]] = k
            raw_col_list = column.tolist()
            col_list = [i.replace(' ', '') for i in raw_col_list]
            for key, value in col_dc.items(): 
                for k in range(len(col_list)): col_list[k] = value if col_list[k]  == key else col_list[k]
            col_ser = pd.Series(col_list, name= f'{name_of_col}_id')
            return col_ser
        else:
            print('this is not pandas series')
            print(type(column))

#this does not work! fix!
def minmax(column):
    try:
        for i in column.tolist(): float(i)
        col = (column - column.min())/(column.max() - column.min())
        return col
    except:
        print('cannot normalize it (string)')
        return column

def num_check(column):
    try:
        for i in column.tolist(): float(i)
        return True
    except:
        return False

data = pd.DataFrame()


#caregorial into indexes
for i in column_ls: data = pd.concat((data, pd.Series(indexer(raw_data[i], i), name = i)), axis = 1)
#normalization
numeric_ls = [i for i in column_ls if num_check(raw_data[i])]
for i in column_ls: data[i] = minmax(raw_data[i]) if num_check(raw_data[i]) else (minmax(data[i]) if i == 'Kcat' else data[i])
print(data.head(20))

