<<<<<<< HEAD
# this programme is to find max length of input
=======
# this programme is to normalize data of each flight
>>>>>>> fe86018530acd348bf451f5f14098bec1a929afc

import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# pre-define value
Fault_types = ['Fault_Fan','Fault_HPC','Fault_HPT','Fault_LPC','Fault_LPT',
               'Nominal_Fan','Nominal_HPC','Nominal_HPT','Nominal_LPC','Nominal_LPT']
title = ['time','alt','MN','TRA','Wf','Fn','SmHPC','SmLPC','SmFan','T48','T2',
         'T24','T30','T50','P2','P15','P30','Nf','Nc','epr','phi','Ps30','NfR','NcR',
         'BPR','farB','htBleed','PCNfRdmd','W31','W32']

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

# func for change str flight_itr
def flight_itr_tostr(itr):
    if itr <= 9:
        return '00'+str(itr)
    elif itr >9 and itr <= 99:
        return '0'+str(itr)
    else:
        return str(itr)

# save fistly all flight data(of course we need smpling 5000 point each flight is too large)
# save norme / mean information  in Data_AllFlight_WithoutNorme.csv
# and also create file for each flight with sampling

sampling_rate = 50
possible_largest_data_length = 0

with open('../Data_strcture.csv') as general_info:

    # read general information
    reader_data_strcture = csv.reader(general_info)
    rows_data_strcture = [row for row in reader_data_strcture]
    rows_data_strcture = rows_data_strcture[1:]

    # open each flight info csv file and excurate scaling process
    for row in rows_data_strcture:
        engine_allflight_datalength = 0
        flight_itr = int(row[3])
        for i in range(flight_itr):
            flight = flight_itr_tostr(i + 1)
            print 'load file//fault_type:' + row[1] + '//engine:' + row[2] + '//flight:' + str(i+1) + '   ---  OK'
            with open('/home/liuyingqi/Desktop/CMAPSS_dataset/' + row[1] + '/Engine0' + row[2] +
                              '/Flight' + flight + '_norme.csv') as flight_info:
                reader_data_flight = csv.reader(flight_info)

                # process to find length
                rows_flight = [ row_data_flight for row_data_flight in reader_data_flight]
                rows_flight = rows_flight[1:]
                row_length = len(rows_flight)

                # compare to find largest possible datalength
                if row_length > possible_largest_data_length:
                    possible_largest_data_length = row_length

print '..............................................'
print 'largest possible data length is' + str(possible_largest_data_length)

# so we find result is 100
