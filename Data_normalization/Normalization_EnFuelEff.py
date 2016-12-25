# this programme is to normalize data of each flight's engine fuel flow and fuel efficiency

import csv
import numpy as np
from sklearn import preprocessing

# load general info from Data_strcture.csv file
with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    Nominal_info = [ row[0] for row in reader]

with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    FaultType_info = [ row[1] for row in reader]

with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    Engine_num = [row[2] for row in reader]

with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    Flight_num = [row[3] for row in reader]

# save fistly all flight data(of course we need smpling 5000 point each flight is too large)
# save norme / mean information  in Data_AllFlight_WithoutNorme.csv
# and also create file for each flight with sampling

with open('../Data_strcture.csv') as general_info:
    # read general information
    reader_data_strcture = csv.reader(general_info)
    rows_data_strcture = [row for row in reader_data_strcture]
    rows_data_strcture = rows_data_strcture[1:]

    # calculate first mean and standard-deviation
    with open('/home/liuyingqi/Desktop/CMAPSS_dataset/Fault_Fan/Engine01/Engine_Fuel_Effic.csv') as flight_info:
        reader_data_flight = csv.reader(flight_info)

        # process change string to float to prepare normalize
        rows_flight = [row_data_flight for row_data_flight in reader_data_flight]
        rows_flight = rows_flight[1:]
        row_length = len(rows_flight[0])
        for rows_flight_row in range(len(rows_flight)):
            for item in range(row_length):
                rows_flight[rows_flight_row][item] = float(rows_flight[rows_flight_row][item])

        # sampling data
        array_to_norme = np.array(rows_flight)
        array_sampling_row = len(array_to_norme[:, 0])
        array_sampling_colunme = len(array_to_norme[0, :])
        array_sampling = np.zeros([array_sampling_row, array_sampling_colunme], dtype=float)
        for i in range(array_sampling_row):
            for j in range(array_sampling_colunme):
                array_sampling [i][j] = array_to_norme[i][j]
        # define standard scaler
        scaler = preprocessing.StandardScaler()
        scaler.fit(array_sampling)

        print 'mean : ------------------------------------'
        print scaler.mean_
        print 'standard-derivation : -------------------------------'
        print scaler.std_
        print 'save -------  down'

    # open each flight info csv file
    for row in rows_data_strcture:
            print 'load file//fault_type:' + row[1] + '//engine:' + row[2] + ' ---  OK'
            with open('/home/liuyingqi/Desktop/CMAPSS_dataset/' + row[1] + '/Engine0' + row[2] + '/Engine_Fuel_Effic.csv') as flight_info:
                reader_data_flight = csv.reader(flight_info)
                # process change string to float to prepare normalize
                rows_flight = [ row_data_flight for row_data_flight in reader_data_flight]
                rows_flight = rows_flight[1:]
                row_length = len(rows_flight[0])
                for rows_flight_row in range(len(rows_flight)):
                    for item in range(row_length):
                        rows_flight[rows_flight_row][item] = float(rows_flight[rows_flight_row][item])

                # sampling data
                array_to_norme = np.array(rows_flight)
                array_sampling_row = (len(array_to_norme[:,0]))
                array_sampling_colunme = len(array_to_norme[0,:])
                array_sampling = np.zeros([array_sampling_row,array_sampling_colunme],dtype=float)
                for i in range(array_sampling_row):
                    for j in range(array_sampling_colunme):
                        array_sampling[i][j] = float(rows_flight[i][j])

                # normalize data based on sampling info
                array_norme = scaler.transform(array_sampling)

                # save normalized data into csv file
                np.savetxt('/home/liuyingqi/Desktop/CMAPSS_dataset/' + row[1] + '/Engine0' + row[2] +
                              '/Engine_Fuel_Effic_norme.csv',array_norme,delimiter=',')

                print 'save -------  down'



