# -*- coding: utf-8 -*-
"""
I'll rewrite it in .py and add a bit later
"""
# satisfying requirements
from pip._internal import main

packages = ('xlsxwriter',
            'pandas',
            'numpy',
            'seaborn',
            'sklearn',
            'openpyxl',
            'matplotlib')
for package in packages:
    try:
        __import__(package)
    except ImportError:
        main(['install', package.split()[0]])

import os
import pandas as pd
import numpy as np
raw_df = pd.read_excel('https://raw.githubusercontent.com/kshiroky/DataCon/main/Database_1.xlsx')

#replace incorrect values
raw_df[raw_df['Type: Organic (O)/inorganic (I)'] == 0] = 'O'
raw_df[raw_df['Type: Organic (O)/inorganic (I)'] == 1] = 'I'

#clear  Nanoparticles
raw_df.Nanoparticle[raw_df.Nanoparticle == 'Copper oxide'] = 'CuO'
raw_df.Nanoparticle[raw_df.Nanoparticle == 'Zinc oxide'] = 'ZnO'

#clear coating
raw_df[raw_df['coat'] == 'Hyaluronic acid '] = 'Hyaluronic acid'
raw_df['coat'] = raw_df['coat'].fillna(0)

#drop useless data of O O O O O
drop_o = raw_df[raw_df.Nanoparticle == 'O'].index
raw_df = raw_df.drop(drop_o)

#drop another useless data
raw_df = raw_df.drop(raw_df[raw_df['Diameter (nm)'] == 'Hyaluronic acid'].index)

#drop not float values
drop_diam = []
for i in raw_df['Concentration μM'].unique():
    try:
        float(i)
    except:
        drop_diam.append(raw_df[raw_df['Concentration μM'] == i].index)

#fill NaN to replace them in the future (by the surface charge of particles)
raw_df['Zeta potential (mV)'] = raw_df['Zeta potential (mV)'].fillna('?')

#replace encoding features
r_ls = raw_df.Cells.tolist()
g_ls = [i.replace('\xad', '-') for i in r_ls]
n_ls = [i.replace('\xa0', '') for i in g_ls]
raw_df.Cells = n_ls
raw_df.Cells.unique()

#again fill NaN
raw_df['Cell line (L)/primary cells (P)'] = raw_df['Cell line (L)/primary cells (P)'].fillna('?')

#replace identical values
raw_df[raw_df['Animal?'] == 'Mice'] = 'Mouse'
raw_df[raw_df['Animal?'] == 'rat'] = 'Rat'
#fill NaN with "Human"
raw_df['Animal?'] = raw_df['Animal?'].fillna('Human')
r_an = raw_df['Animal?'].to_list()
check_an = raw_df['Human(H)/Animal(A) cells'].tolist()
#replace real NaN with ?
for i in range(len(r_an)): r_an[i] = r_an[i].replace('Human', '?')  if not check_an[i] == 'H' else r_an[i]
raw_df['Animal?'] = r_an
raw_df['Animal?'].unique()

#delete useless data
drop_cm = [i for i in raw_df[raw_df['Cell morphology'] == 'Rat'].index]
raw_df = raw_df.drop(drop_cm)
drop_cm = [i for i in raw_df[raw_df['Cell morphology'] == 'Mouse'].index]
raw_df = raw_df.drop(drop_cm)

#fill NaN with the closest neighbour's value
raw_df['Cell morphology'] = raw_df['Cell morphology'].fillna(method= 'bfill', limit = 1)

#fill NaN with the closest neighbour's value
raw_df['Cell age: embryonic (E), Adult (A)'] = raw_df['Cell age: embryonic (E), Adult (A)'].fillna(method= 'bfill', limit= 1)

tiss_ls = raw_df['Cell-organ/tissue source'].to_list()
n_ts_ls = [i.lower() for i in tiss_ls]
raw_df['Cell-organ/tissue source'] = n_ts_ls
raw_df['Cell-organ/tissue source'].unique()

#encoding features again 
raw_df['Test'] = raw_df['Test'].fillna('?')
ls = raw_df['Test'].to_list()
n_ls = [i.replace('\xad', '-') for i in ls]
raw_df['Test'] = n_ls
raw_df['Test'].unique()

#some functions:
#turns the column into lower register list
def low(column):
    ls = column.tolist()
    n_ls = [i.lower() for i in ls]
    return n_ls

#replaces encoded dashes with real ones
def rm_xad(column):
    ls = column.tolist()
    n_ls = [i.replace('\xad', '-') for i in ls]
    return n_ls

#fill NaN with the closest neighbour's value
raw_df['Test indicator'] = raw_df['Test indicator'].fillna(method= 'bfill', limit = 1)
#for escaping problems with different registers of the same words
raw_df['Test indicator'] = low(raw_df['Test indicator'])

#fill NaN with the closest neighbour's value
raw_df['Biochemical metric'] = raw_df['Biochemical metric'].fillna(method = 'bfill', limit = 1)

#the list of df headers
col_ls = raw_df.columns
col_ls

#separate unreal cell viability to another df
neg_index = raw_df[raw_df['% Cell viability'] < 0]
pos_index = raw_df[raw_df['% Cell viability'] > 100]
predict_data = neg_index.append(pos_index)
predict_data

#data with correct viability
data = raw_df.drop(predict_data.index)

#data.to_excel(str(os.cwd()+'/processed_data.xlsx')

import xlsxwriter

# Подключение таблицы
file = 'https://raw.githubusercontent.com/kshiroky/DataCon/main/Database_2.xlsx'

# Объявление массивов с заголовками двух листов
headers1 = []
headers2 = []


output = []
with pd.ExcelFile(file) as xls:
    # Запись в переменную первого листа
    df1 = pd.read_excel(xls, sheet_name='Лист1')
    for i in df1:
        headers1.append(i)
    # Запись в перменную второго листа
    df2 = pd.read_excel(xls, sheet_name='Лист2')
    for i in df2:
        headers2.append(i)
xls.close()

# Заголовки первой таблицы второго листа
headers2_1 = ['Material type', 'Elements', 'Number of atoms', 'Molecular weight (g/mol)',
              'Topological polar surface area (Å²)', 'Unit Cell Parameters', 'Density (g/cm3)',
              'Electronegativity', 'Ionic radius', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
              'Unnamed: 10', 'Unnamed: 11']
list2_1 = {}

# Чтение первой таблицы второго листа
for i in range(len(headers2_1)):
    for j in range(16):
        output.append(df2[headers2_1[i]][j])
    list2_1[headers2_1[i]] = output
    output = []

# Заголовки второй таблицы второго листа
headers2_2 = ['Material type', 'Elements', 'Molecular weight (g/mol)']
ListHeaders2_2 = {headers2_2[0]: 'Material type',
                  headers2_2[1]: 'Элемент',
                  headers2_2[2]: 'Ионный радиус (пм)'}

list2_2 = {}

# Чтение второй таблицы второго листа
for i in range(3):
    for j in range(16):
        output.append(df2[headers2_2[i]][j+19])
    if headers2_2[i] == 'Molecular weight (g/mol)':
        list2_2['Ionic radius'] = output
    else:
        list2_2[headers2_2[i]] = output
    output = []

# Заголовки третьей таблицы второго листа
list2_3 = {}
headers2_3 = ['Material type', 'Elements']

# Чтение третьей таблицы второго листа
for i in range(2):
    for j in range(93):
        output.append(df2[headers2_3[i]][j+38])
    if headers2_3[i] == 'Material type':
        list2_3['Elements'] = output
    else:
        list2_3['Electronegativity'] = output
    output = []

# Чтение первого листа
list1 = {}
for i in range(len(headers1)):
    for j in range(1068):
        output.append(df1[headers1[i]][j])
    list1[headers1[i]] = output
    output = []

# 0 0 Material type
# 0 1 Elements
# 0 2 Electronegativity
# 0 3 Ionic radius
# 0 4 Core size (nm)
# 0 5 Hydro size (nm)
# 0 6 Surface charge (mV)
# 0 7 Surface area (m2/g)
# 0 8 Cell type
# 0 9 Exposure dose (ug/mL)
# 0 10 Number of atoms
# 0 11 Molecular weight (g/mol)
# 0 12 Topological polar surface area (Å²)
# 0 13 a (Å)
# 0 14 b (Å)
# 0 15 c (Å)
# 0 16 α (°)
# 0 17 β (°)
# 0 18 γ (°)
# 0 19 Density (g/cm3)
# 0 20 Viability (%)

# Массив с данными о заголовках первой таблицы второго листа
listGet2_1 = ['Material type', 'Elements', 'Number of atoms', 'Molecular weight (g/mol)',
            'Topological polar surface area (Å²)', 'Unit Cell Parameters', 'Density (g/cm3)',
             'Electronegativity', 'Ionic radius', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
              'Unnamed: 10', 'Unnamed: 11']

# Массив с данными о заголовках второй таблицы второго листа
listGet2_2 = ['Material type', 'Elements', 'Ionic radius']

# Массив с данными о заголовках третьей таблицы второго листа
listGet2_3 = ['Elements', 'Electronegativity']

# Массив с данными о заголовках таблицы первого листа
headers3 = ['Material type', 'Elements', 'Electronegativity', 'Ionic radius', 'Core size (nm)', 'Hydro size (nm)',
            'Surface charge (mV)', 'Surface area (m2/g)', 'Cell type', 'Exposure dose (ug/mL)', 'Number of atoms',
            'Molecular weight (g/mol)', 'Topological polar surface area (Å²)', 'a (Å)', 'b (Å)', 'c (Å)',
            'α (°)', 'β (°)', 'γ (°)', 'Density (g/cm3)', 'Viability (%)']

# Словарь для определения индекса заголовка
headers3List = {'Material type': 0,
                'Elements': 1,
                'Electronegativity': 2,
               'Ionic radius': 3,
               'Core size (nm)': 4,
               'Hydro size (nm)': 5,
            'Surface charge (mV)': 6,
               'Surface area (m2/g)': 7,
               'Cell type': 8,
               'Exposure dose (ug/mL)': 9,
               'Number of atoms': 10,
            'Molecular weight (g/mol)': 11,
               'Topological polar surface area (Å²)': 12,
               'a (Å)': 13, 'b (Å)': 14, 'c (Å)': 15,
            'α (°)': 16, 'β (°)': 17, 'γ (°)': 18,
               'Density (g/cm3)': 19,
               'Viability (%)': 20,
                'Unit Cell Parameters': 13,
                'Unnamed: 7': 14,
                'Unnamed: 8': 15,
                'Unnamed: 9': 16,
                'Unnamed: 10': 17,
                'Unnamed: 11': 18
                }
import pandas as pd
import matplotlib.pyplot as plt

# Объявление директории файла
file = 'https://raw.githubusercontent.com/kshiroky/DataCon/main/Database_1.xlsx'

# Массив с заголовками
columns = ['Nanoparticle', 'Type: Organic (O)/inorganic (I)', 'coat', 'Diameter (nm)', 'Concentration μM', 'Zeta potential (mV)',
                   'Cells', 'Cell line (L)/primary cells (P)', 'Human(H)/Animal(A) cells', 'Animal?', 'Cell morphology',
                   'Cell age: embryonic (E), Adult (A)', 'Cell-organ/tissue source', 'Exposure time (h)', 'Test',
                   'Biochemical metric', '% Cell viability', 'Interference checked (Y/N)',
                   'Colloidal stability checked (Y/N)', 'Positive control (Y/N)']
Nanoparticle = []
Coat = []
Diameter = []
HA = []

# Чтение таблицы и цикл обработки значений и исключений, счетчик значений
with pd.ExcelFile(file) as xls:
    df = pd.read_excel(xls, sheet_name='Master sheet')
    result = {}
    for i in columns:
        try:
            name = (df[i])
            name_types = {}
            for j in name:
                if i == 'Nanoparticle':
                    Nanoparticle.append(j)
                elif i == 'coat' and str(j) != 'nan':
                    if j == 'Digestive enzymes':
                        Coat.append('DE')
                    elif j == 'simethicone then esters on top':
                        Coat.append('simethicone')
                    else:
                        Coat.append(j)
                elif i == 'Diameter (nm)':
                    Diameter.append(j)
                elif i == 'Human(H)/Animal(A) cells':
                    if j == 'H':
                        HA.append('Human')
                    else:
                        HA.append('Animal')

                if j in name_types:
                    name_types[j] += 1
                elif j == 'Copper Oxide':
                    name_types['CuO'] += 1
                elif j == 'Zinc oxide':
                    name_types['ZnO'] += 1
                elif j == 'Iron oxide':
                    if 'FeO' not in name_types:
                        name_types['FeO'] = 1
                    else:
                        name_types['FeO'] += 1
                elif j == 'Hydroxyapatite':
                    if 'Hydroxyapatite' not in name_types:
                        name_types['Ca10(PO4)6(OH)2'] = 1
                    else:
                        name_types['Ca10(PO4)6(OH)2'] += 1
                elif j == 'Eudragit RL':
                    if 'Eudragit RL' not in name_types:
                        name_types['C11H21NO4'] = 1
                    else:
                        name_types['C11H21NO4'] += 1
                else:
                    name_types[j] = 1
        except:
            pass
        result[i] = name_types
    with open('result.txt', 'w') as f:
        f.write(str(result))


# # Построение графиков
# График элементов Nanoparticle
fig = plt.gcf()
plt.hist(Nanoparticle, edgecolor='black', bins=100)
plt.xlabel('Values')
plt.ylabel('Frequencies')
fig.set_size_inches(30, 9)
fig.suptitle("Nanoparticle", fontsize=16)
plt.xticks(rotation=65)
plt.show()

# График элементов Coat
fig = plt.gcf()
plt.hist(Coat, edgecolor='black', bins=100)
plt.xlabel('Values')
plt.ylabel('Frequencies')
fig.set_size_inches(30, 9)
fig.suptitle("Coat", fontsize=16)
plt.xticks(rotation=80)
plt.show()

# График элементов Diameter
fig = plt.gcf()
plt.hist(Diameter, edgecolor='black', bins=100)
plt.xlabel('Values')
plt.ylabel('Frequencies')
fig.set_size_inches(30, 9)
fig.suptitle("Diameter", fontsize=16)
plt.xticks(rotation=80)
plt.show()

# График элементов Human(H)/Animal(A) cells
fig = plt.gcf()
plt.hist(HA, edgecolor='black', bins=3)
plt.xlabel('Values')
plt.ylabel('Frequencies')
fig.set_size_inches(30, 9)
fig.suptitle("Human(H)/Animal(A) cells", fontsize=16)
plt.show()

# Подключение файла с общей таблицей
workbook = xlsxwriter.Workbook('Result.xlsx')
worksheet = workbook.add_worksheet()
# Запись первого листа
for i in range(len(headers3)):
    worksheet.write(0, i, headers3[i])
    for j in range(len(list1[headers3[i]])):
        worksheet.write(j+1, i, str(list1[headers3[i]][j]))

# Запись первой таблицы второго листа
for i in range(len(listGet2_1)):
    for j in range(len(list2_1[listGet2_1[i]])-1):
        worksheet.write(j+1070, headers3List[listGet2_1[i]], str(list2_1[listGet2_1[i]][j+1]))

# Запись второй таблицы второго листа
for i in range(len(listGet2_2)):
    for j in range(len(list2_2[listGet2_2[i]])):
        worksheet.write(j+1086, headers3List[listGet2_2[i]], str(list2_2[listGet2_2[i]][j]))

# Запись третьей таблицы второго листа
for i in range(len(listGet2_3)):
    for j in range(len(list2_3[listGet2_3[i]])):
        worksheet.write(j+1103, headers3List[listGet2_3[i]], str(list2_3[listGet2_3[i]][j]))

workbook.close()


