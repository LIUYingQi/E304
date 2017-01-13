import csv
from matplotlib import pyplot
import pandas as pd

with open('../Data_strcture.csv','rb') as csvfile:
    info = pd.read_csv(csvfile)
    info = info.values
    # print info
    pyplot.figure(figsize=(15,7))

    file = '/home/liuyingqi/Desktop/CMAPSS_dataset/Fault_HPC/Engine02/EngineHealth.csv'
    print file
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        column = [ row[0] for row in reader]
        pyplot.subplot(121)
        pyplot.plot(column[1:])
        pyplot.title('case : fault occured at HPC',y=1.04,fontsize=16)
        pyplot.ylabel('HI',fontsize=16)
        pyplot.xlabel('Flight number',fontsize=16)
        pyplot.ylim([0, 1])

    file = '/home/liuyingqi/Desktop/CMAPSS_dataset/Nominal_HPC/Engine03/EngineHealth.csv'
    print file
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        column = [row[0] for row in reader]
        pyplot.subplot(122)
        pyplot.plot(column[1:])
        pyplot.title('case : no fault', y=1.04,fontsize=16)
        pyplot.ylabel('HI',fontsize=16)
        pyplot.xlabel('Flight number',fontsize=16)
        pyplot.ylim([0, 1])
        pyplot.show()


