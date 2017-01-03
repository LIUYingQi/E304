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

with open('/home/liuyingqi/Desktop/CMAPSS_dataset/'+Fault_type+'/Engine0'+str(Engine)+'/Flight0'+str(int(Fault_flight))+'.csv') as csvfile:

    reader = csv.reader(csvfile)
    reader.next()
    column = [ row for row in reader]
    column = np.array(column)
    pyplot.figure(figsize=(10,30))
    pyplot.suptitle('A flight condition example (randomly selected from dataset)')
    pyplot.subplots_adjust(bottom = 0.05,right = 0.9,top=0.9,wspace = 0.4,hspace= 0.3)
    pyplot.subplot(3,2,1)
    pyplot.plot(column[:,0],column[:,0])
    pyplot.title('time')
    pyplot.xlabel('sec')
    pyplot.ylabel('sec')
    pyplot.subplot(3,2,2)
    pyplot.plot(column[:,0],column[:,1])
    pyplot.ylim(0,40000)
    pyplot.title('Altitude')
    pyplot.ylabel('ft')
    pyplot.xlabel('sec')
    pyplot.subplot(3,2,3)
    pyplot.plot(column[:,0],column[:,2])
    pyplot.ylim(0,1)
    pyplot.title('Mach number')
    pyplot.ylabel('MN')
    pyplot.xlabel('sec')
    pyplot.subplot(3,2,4)
    pyplot.plot(column[:,0],column[:,3])
    pyplot.ylim(0,120)
    pyplot.title('Throttle resolver angle')
    pyplot.ylabel('deg')
    pyplot.xlabel('sec')
    pyplot.subplot(3,2,5)
    pyplot.plot(column[:,0],column[:,4])
    pyplot.title('Fuel flow')
    pyplot.ylabel('pps')
    pyplot.xlabel('sec')
    pyplot.subplot(3,2,6)
    pyplot.plot(column[:,0],column[:,5])
    pyplot.title('Net thrust')
    pyplot.ylabel('lbf')
    pyplot.xlabel('sec')
    pyplot.show()