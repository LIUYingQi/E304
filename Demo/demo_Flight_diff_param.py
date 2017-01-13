import csv
from matplotlib import pyplot
import numpy as np
import pandas as pd

Fault_type = 'Fault_LPC'
Engine = 2
sampling_rate = 20

info = pd.read_csv('../Data_strcture.csv',dtype={'Engine_num':np.int8})
info = info.loc[info.FaultType_info == Fault_type ]
info = info.loc[info.Engine_num == Engine]
print info
Fault_flight = info.Fault_flight
Fault_time = info.Fault_time/20.0 -7
print Fault_time
print '/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'_diff_norme.csv'

titleset = ['f.d alt','f.d MN','f.d TRA','f.d Wf',
            'f.d Fn','f.d SmHPC','f.d SmLPC','f.d SmFan',
            'f.d T48','f.d T2','f.d T24','f.d T30'
    ,'f.d T50','f.d P2','f.d P15','f.d P30','f.d Nf','f.d Nc','f.d epr',
            'f.d phi','f.d Ps30','f.d NfR','f.d NcR','f.d BPR','f.d farB',
            'f.d htBleed','f.d PCNfRdmd','f.d W31','f.d W32']


with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'_diff_norme.csv') as csvfile:

    reader = csv.reader(csvfile)
    reader.next()
    column = [ row for row in reader]
    column = np.array(column)
    pyplot.figure(figsize=(8, 10))
    # pyplot.title('A flight\'s all parameters after standardization and detrending (randomly selected from dataset)', y=1.04)
    for i in range(29):
        pyplot.plot(column[:,i],label=titleset[i])
        # pyplot.legend(str(i))
    # pyplot.vlines(Fault_time, -6, 6, 'red')
    pyplot.xlim(-5, 260)
    pyplot.xlabel('sampling points, Fs=20Hz', fontsize=18)
    pyplot.ylabel('standardized forward difference parameters', fontsize=18)
    pyplot.text(20, -3, 'take-off stage',fontsize=18)
    pyplot.text(90, -3, 'cruise stage',fontsize=18)
    pyplot.text(160, -3, 'descend stage',fontsize=18)
    pyplot.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    pyplot.show()