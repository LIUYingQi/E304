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
Fault_flight = info.Fault_flight
Fault_time = info.Fault_time
print Fault_time
Fault_time = float(Fault_time)/float(20) -1

titleset = ['alt','MN','TRA','Wf','Fn','SmHPC','SmLPC','SmFan','T48','T2','T24','T30'
    ,'T50','P2','P15','P30','Nf','Nc','epr','phi','Ps30','NfR','NcR','BPR','farB','htBleed','PCNfRdmd','W31','W32']

with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'_norme.csv') as csvfile:

    reader = csv.reader(csvfile)
    reader.next()
    column = [ row for row in reader]
    column = np.array(column)
    pyplot.figure(figsize=(10,10))
    # pyplot.title('A flight\'s all parameters after standardization (randomly selected from dataset)',y=1.04)
    for i in range(29):
        pyplot.plot(column[:,i],label=titleset[i])
        # pyplot.legend(str(i))
        # pyplot.axhline()
        # pyplot.axvline()
        pyplot.xlim(-5,260)
        pyplot.xlabel('sampling points, Fs=20Hz',fontsize=18)
        pyplot.ylabel('standardized parameters',fontsize=18)
        pyplot.text(15,-3,'take-off stage',fontsize=16)
        pyplot.text(100,-3,'cruise stage',fontsize=16)
        pyplot.text(180,-3, 'descend stage',fontsize=16)
        pyplot.legend(bbox_to_anchor=(1.01,1),loc=2,borderaxespad=0.)
    pyplot.show()