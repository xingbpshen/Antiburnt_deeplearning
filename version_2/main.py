from version_2.Modules import antiburn_modules
import pandas as pd
import numpy as np


""" Test the antiburn module
"""
am = antiburn_modules.AntiburnModules('/Users/AntonioShen')
df = pd.read_csv('/Users/AntonioShen/MyTestData/IR03-5.csv', encoding='latin-1', skiprows=121, usecols=[0, 1, 2])
ds = df.values
arr = np.array(ds)
for i in range(0, len(arr) - 60):
    a = []
    for j in range(i, i + 60):
        a.append(arr[j])
    a = np.array(a)
    a = np.reshape(a, (60, 3))
    flag = am.antiburnPredict(a)
    print(flag)

exit()
