import csv
from matplotlib import pyplot
from scipy import stats
from scipy import signal
import numpy as np
import pandas as pd

# to find
fault_type = 'Fault_HPT'
engine = '2'
mesurement = 'TRA'

# func for change str flight_itr
def flight_itr_tostr(itr):
    if itr <= 9:
        return '00' + str(itr)
    elif itr > 9 and itr <= 99:
        return '0' + str(itr)
    else:
        return str(itr)

# load general info from Data_strcture.csv file
with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    reader.next()
    data_rows = [row for row in reader]
    for i in range(len(data_rows)):
        if(data_rows[i][1]==fault_type and data_rows[i][2]==engine):
            flight_num = data_rows[i][3]
            break
    general_info.close()

# calculate corr
corr_alt = []
corr_MN = []
corr_TRA = []
corr_Wf = []
corr_Fn = []

for flight_item in range(int(flight_num)):

    with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+fault_type+'/Engine0'+engine+'/Flight'+flight_itr_tostr(flight_item+1)+'.csv') as csvfile:
        info = pd.read_csv(csvfile,engine='c')
        a = stats.pearsonr(info['T48'],info[mesurement])
        corr_alt.append(a[0])
        a = stats.pearsonr(info['T2'], info[mesurement])
        corr_MN.append(a[0])
        a = stats.pearsonr(info['T24'], info[mesurement])
        corr_TRA.append(a[0])
        a = stats.pearsonr(info['T30'], info[mesurement])
        corr_Wf.append(a[0])
        a = stats.pearsonr(info['T50'], info[mesurement])
        corr_Fn.append(a[0])
        csvfile.close()

pyplot.figure()
pyplot.plot(corr_Fn)
pyplot.plot(corr_Wf)
pyplot.plot(corr_TRA)
pyplot.plot(corr_MN)
pyplot.plot(corr_alt)
pyplot.plot(signal.savgol_filter(corr_Fn,11,7))
pyplot.plot(signal.savgol_filter(corr_Wf,11,7))
pyplot.plot(signal.savgol_filter(corr_TRA,11,7))
pyplot.plot(signal.savgol_filter(corr_MN,11,7))
pyplot.plot(signal.savgol_filter(corr_alt,11,7))
pyplot.legend(('T48','T2','T24','T30','T50','T48sg','T2sg','T24sg','T30sg','T50sg'))
pyplot.show()
