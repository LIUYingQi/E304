# this file is to add a normalization of life cycle info

import csv
import numpy as np
from sklearn import preprocessing

with open('../Data_strcture.csv') as general_info:
    # read general information
    reader_data_strcture = csv.reader(general_info)
    rows_data_strcture = [row for row in reader_data_strcture]
    rows_data_strcture = rows_data_strcture[1:]

    # calculate first mean and standard-deviation
    flight_num = int(rows_data_strcture[0][3])
    print 'Fault fan flight num' + str(flight_num)
    flight_num = np.linspace(1, flight_num, flight_num)
    print flight_num.shape
    print flight_num

    scaler = preprocessing.StandardScaler()
    scaler.fit(flight_num)
    print 'mean : ------------------------------------'
    print scaler.mean_
    print 'standard-derivation : -------------------------------'
    print scaler.std_
    print 'save -------  down'

    # open each flight info csv file
    for row in rows_data_strcture:
        flight_num = int(row[3])
        print 'Fault fan flight num' + str(flight_num)
        flight_num = np.linspace(1, flight_num, flight_num)
        print flight_num

        # normalize data based on sampling info
        array_norme = scaler.transform(flight_num)

        # save normalized data into csv file
        np.savetxt('/home/liuyingqi/Desktop/CMAPSS_dataset/' + row[1] + '/Engine0' + row[2] +
                   '/Lifecycle_norme.csv',array_norme,delimiter=',')

        print 'save -------  down'



