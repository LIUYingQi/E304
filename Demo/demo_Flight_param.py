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

with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'_norme.csv') as csvfile:

    reader = csv.reader(csvfile)
    reader.next()
    column = [ row for row in reader]
    column = np.array(column)
    for i in range(29):
        pyplot.plot(column[:,i])
        pyplot.legend(str(i))
        pyplot.axhline()
        pyplot.axvline()
        pyplot.vlines(Fault_time,-6,6,'red')
        pyplot.show()