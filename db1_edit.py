import pandas as pd
import numpy as np 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor as forestreg
from sklearn.model_selection import cross_validate
#get the path pf the db1
name = 'Database_1.xlsx'
path = os.path.realpath(name)

#function which turns categorial values into indexes:
def indexer(column, name_of_col = 'smth'):
    try: 
        for i in column.tolist(): float(i)
        pass
    except:
        if type(column) == type(pd.Series()):
            col_dc = {}
            uniq_ls = column.unique()
            for k in range(len(uniq_ls)): col_dc[uniq_ls[k]] = k
            col_list = column.tolist()
            for key, value in col_dc.items(): 
                for k in range(len(col_list)): col_list[k] = value if col_list[k]  == key else col_list[k]
            col_ser = pd.Series(col_list, name= f'{name_of_col}_id')
            return col_ser
        else:
            print('this is not pandas series')
            print(type(column))

#for missed values
knn = KNNImputer(n_neighbors = 5, weights = 'distance')

#class for column proccesing 
class my_col:
    def __init__(self, column):
        if type(self) == type(pd.Series()):
            self.ls = column.tolist()
    #remove spaces in the begging and in the end of the cell
    def stripper(self):
        try:
            for i in self.ls: float(i)
            pass
        except:
            n_ls = [i.strip() for i in self.ls]
            return n_ls
    #replace \xad & \xa0 with - and nothing
    def decoding(self):
        try:
            for i in self.ls: float(i)
            pass
        except:
            n_ls = [i.replace('\xad', '-') for i in self.ls]
            n_ls = (i.replace('\xa0', '') for i in n_ls)
            return n_ls
    #missed values processing 
    def nan_proc(self):
        if self.isnull().any():
            return self.fillna(knn.fit_transform(self))




#create df
raw_df = pd.read_excel(path)
#get columns names of df
columns = raw_df.columns

