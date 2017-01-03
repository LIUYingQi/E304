Step_num = 100

# load data
path = '../Graphe_result_saved/Model' + str(Model_num) + '/File' + str(File_num + 1) + '/Step' + str(
    Step_num) + '_prediction_RUL.pkl'
with open(path, 'rb') as test_result:
    predict_RUL = pickle.load(test_result)

predict_RUL = np.array(predict_RUL, dtype=np.float32)
print ' --------------------- loading prediction RUL ------------------- '
print ' prediction for RUL : '
print predict_RUL

# load real RUL
print ' --------------------- loading real RUL ------------------- '
if File_num + 1 > 10:
    print ' case : no fault '
    with open('../Data_strcture.csv') as general_info:
        reader = csv.reader(general_info)
        data_rows = [row for row in reader]
        flight_num = int(data_rows[File_num + 1][3])
        print 'flight : ' + str(flight_num)
        real_RUL = np.arange(flight_num, 0, -1) - 1
else:
    print ' case : with fault '
    with open('../Data_strcture.csv') as general_info:
        reader = csv.reader(general_info)
        data_rows = [row for row in reader]
        flight_num = int(data_rows[File_num + 1][3])
        print ' flight : ' + str(flight_num)
        fault_flight = int(data_rows[File_num + 1][4])
        real_RUL = np.append(np.ones(fault_flight - 1) * (flight_num - fault_flight + 1),
                             np.arange(flight_num - fault_flight + 1, 0, -1) - 1)

# define : pre-plot
step = np.arange(flight_num) + 1
step_prediction = np.arange(1, flight_num + 1)

# plot
fig1 = plt.figure('fig1', figsize=(10, 10))
plt.plot(step, real_RUL)
plt.plot(step_prediction, predict_RUL)
plt.plot(step_prediction, predict_RUL - real_RUL[-len(predict_RUL):])
plt.xlabel('step ')
plt.ylabel('mean difference prediction and real RUL')
plt.title('mean for difference between prediction RUL and real RUL')
plt.show()