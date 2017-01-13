import csv
from matplotlib import pyplot
from matplotlib import patches
import pandas as pd

# print info
pyplot.figure(figsize=(15,7))
pyplot.title('HI degradation process ( all instances ) ', y=1.04, fontsize=16)
pyplot.ylabel('HI', fontsize=16)
pyplot.xlabel('Flight number', fontsize=16)
pyplot.ylim([0, 1])

file_list = ['Fault_Fan','Fault_HPC','Fault_HPC','Fault_HPT','Fault_HPT','Fault_LPT','Fault_LPT']
engine_num = [2, 1,2, 1,2,      1,2]
patch_red = patches.Patch(color='red',label='fault occurred')

for x,y in zip(file_list,engine_num):

    file = '/home/liuyingqi/Desktop/CMAPSS_dataset/'+x+'/Engine0'+str(y)+'/EngineHealth.csv'
    print file
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        column = [ row[0] for row in reader]
        pyplot.plot(column[1:],color='red')


file_list = ['Nominal_Fan',
             'Nominal_HPC','Nominal_HPC','Nominal_HPC','Nominal_HPC','Nominal_HPC',
             'Nominal_HPT','Nominal_HPT','Nominal_HPT','Nominal_HPT','Nominal_HPT',
             'Nominal_LPC','Nominal_LPC','Nominal_LPC',
             'Nominal_LPT','Nominal_LPT','Nominal_LPT','Nominal_LPT','Nominal_LPT']
engine_num = [1,  1,2,3,4,5,  1,2,3,4,5,  1,3,4,    1,2,3,4,5]
patch_green = patches.Patch(color='green',label='no fault occurred')

for x,y in zip(file_list,engine_num):

    file = '/home/liuyingqi/Desktop/CMAPSS_dataset/'+x+'/Engine0'+str(y)+'/EngineHealth.csv'
    print file
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        column = [ row[0] for row in reader]
        pyplot.plot(column[1:],color='green')
pyplot.legend(handles=[patch_red,patch_green])
pyplot.show()

