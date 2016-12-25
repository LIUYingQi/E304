import csv
from matplotlib import pyplot
import pandas as pd

with open('../Data_strcture.csv','rb') as csvfile:
    info = pd.read_csv(csvfile)
    info = info.values
    # print info
    for row in info:
        file = '/home/liuyingqi/Desktop/CMAPSS_dataset/'+str(row[1])+'/Engine0'+str(row[2])+'/EngineHealth.csv'
        print file
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            column = [ row[0] for row in reader]
            pyplot.plot(column[1:])
            pyplot.show()




