# this programe is to label RUL after each flight

import csv

times = 123
with open('/home/liuyingqi/Desktop/CMAPSS_dataset/Nominal_LPT/Engine05/EngineRUL.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['E_RUL'])
    for i in range(times):
        spamwriter.writerow([times-1-i])