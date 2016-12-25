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
Fault_time = info.Fault_time
Fault_flight = info.Fault_flight
print Fault_time

with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'_diff_norme.csv') as csvfile:

    reader = csv.reader(csvfile)
    reader.next()
    column = [ row for row in reader]
    column = np.array(column)
    for i in range(30):
        pyplot.figure()
        pyplot.plot(column[:,0],column[:,i])
        pyplot.legend(str(i))
        pyplot.scatter(np.around(Fault_time),column[np.around(Fault_time),i],s = 100,c = 'red',alpha=0.3)
        pyplot.show()