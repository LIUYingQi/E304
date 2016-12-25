import numpy as np
import pandas as pd

info = pd.read_csv('../Data_strcture_notendflight.csv',engine='c')
print info
info['RUL_total'] = info['Flight_num']*(info['Flight_num']+1)/2
print info

print info['RUL_total'].groupby(by=info['type']).sum()