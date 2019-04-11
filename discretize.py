
"""
Discretize the continuous variables using the equal frequency intervals 

@author: Marco Benjumeda
"""
import pandas
import numpy as np
from copy import deepcopy


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

# Checks if an array can be converted to float
def is_numeric_array(arr):
    try:
        np.array(arr).astype(np.float)
    except Exception:
        return False
    else:
        return True
    
# Get variables to discretize (Numeric variables with more than 8 values)
def get_vars_2_discretize(df):
    v2d = []
    for i in range(df.shape[1]):
        if  is_numeric_array(df.iloc[:,i].values) and len(np.unique(df.iloc[:,i])) > 8:
            v2d.append(i)
    return(v2d)
            

def get_eq_freq_int(values_in, cutpoints):
    values = deepcopy(values_in)
    for j in range(len(values)):
        if not pandas.isnull(values[j]):
            cat = 0
            while cat<len(cutpoints) and values[j] > cutpoints[cat]:
                cat = cat + 1
            values[j] = cat + 1
    return values
    

def equal_freq_disc(data , q_vars, bins = 3):
    freq = np.ceil(float(data.shape[0])/bins)
    return fixed_freq_disc(data , q_vars, freq)

# Discretizes the quantitative variables using the fixed frequency approach
def fixed_freq_disc(data , q_vars, freq = 30):
    
    data_disc = deepcopy(data)
    cut_list = []
    for i in range(len(q_vars)):
        ncuts = np.floor((data.shape[0] - sum(pandas.isnull(data.iloc[:,q_vars[i]]))) / freq)
        values = deepcopy(data.iloc[:,q_vars[i]].values.astype(np.float))
        disc=pandas.qcut(values,int(ncuts), duplicates='drop')
        cutpoints = [x.right for x in disc.categories][:-1]
        cut_list.append(cutpoints)
        values = get_eq_freq_int(values,cutpoints)
        data_disc.iloc[:,q_vars[i]] = values
    return data_disc, cut_list
    
    
def fixed_freq_disc_test(data , q_vars, cut_list):
    data_disc = deepcopy(data)
    for i in range(len(q_vars)):
        values = get_eq_freq_int(data.iloc[:,q_vars[i]].values.astype(np.float),cut_list[i])
        data_disc.iloc[:,q_vars[i]] = values
    return data_disc
        
        
    

