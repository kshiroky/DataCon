import pandas as pd
import numpy as np 
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

class my_col:
    def __init__(self, column):
        if type(self) == type(pd.Series()):
            self.ls = column.tolist()
    def stripper(self):
        try:
            for i in self.ls: float(i)
        except:
            n_ls = [i.strip() for i in self.ls]
            return n_ls


#create df
raw_df = pd.read_excel(path)
#get columns names of df
columns = raw_df.columns

