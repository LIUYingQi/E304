import csv
from matplotlib import pyplot

with open('/home/liuyingqi/Desktop/CMAPSS_dataset/Fault_Fan/Engine01/Engine_Fuel_Effic.csv') as csvfile:
    reader = csv.reader(csvfile)
    column = [ row[1] for row in reader]
    print column
    print '\n'
    print column[1:]
    pyplot.plot(column[1:])
    pyplot.show()


