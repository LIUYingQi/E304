# this programe is to label fault types after each flight

import csv
import os
fault_type = ''
fault_flight = 1
total_flight = 10

with open('../Data_strcture.csv') as general_info:
    # read general information
    reader_data_strcture = csv.reader(general_info)
    rows_data_strcture = [row for row in reader_data_strcture]
    rows_data_strcture = rows_data_strcture[1:]

    # open each flight info csv file
    for row in rows_data_strcture:
        fault_type = row[1]
        engine_num = row[2]
        if row[4] != 'NaN':
            fault_flight = int(row[4])
        else:
            fault_flight = None
        total_flight = int(row[3])
        with open('/home/liuyingqi/Desktop/CMAPSS_dataset/' + fault_type + '/Engine0'+engine_num+'/EngineFault_type.csv','wb') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['E_FaultTypes'])
            if not fault_flight == None:
                for i in range(fault_flight-1):
                    spamwriter.writerow(['Norminal'])
                for i in range(fault_flight-1,total_flight):
                    spamwriter.writerow([fault_type])
            else:
                for i in range(total_flight):
                    spamwriter.writerow(['Norminal'])
