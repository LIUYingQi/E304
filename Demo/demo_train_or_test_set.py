# this program is to show if it work well for cpickle part to load data

import cPickle as pickle

with open('../Train_data_set/TrainLabel1.pkl','rb') as dataset_info_to_load:
    data =  pickle.load(dataset_info_to_load)

print data.shape
print data[0]
