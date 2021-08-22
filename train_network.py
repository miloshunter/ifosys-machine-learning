import time, os, argparse, io
# Tensorflow and numpy!
import tensorflow.keras.models
from tensorflow.keras import optimizers
import numpy as np
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt
import colour
import colour.plotting as cp
import xlrd
import scipy.interpolate as ip
from numpy import diff

cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
#print(cmfs)
illuminant='D65'
sd_illuminant = colour.ILLUMINANTS_SDS[illuminant]

#cp.plot_multi_sds(cmfs)

illuminant_xy=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]

EPOCHS = 1000
BATCH = 250
#TRAIN_OR_LOAD = "LOAD"
TRAIN = False
LOAD = True
REMOVE_MEASURING_POINTS = False

directory = os.path.dirname(os.path.realpath(__file__))

des = []


def get_ref_sd(data):

    nm = np.linspace(380, 730, 36)  # example new x-axis
    data_plot = dict(zip(nm, data))

    sd = colour.SpectralDistribution(data_plot, name='EyeOnePro')

    return sd


def get_mach_lrn_sd(data):
    nm = np.linspace(380, 730, 36)  # example new x-axis

    data_plot = dict(zip(nm, data))
    sd = colour.SpectralDistribution(data_plot, name='MachineLearning')  # w=np.array(measWhite)

    return sd

if __name__ == '__main__':  # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument

    # We create a SummaryWriter to save data for TensorBoard
    result_folder = directory + '/results/' + str(int(time.time()))
    x_train = []
    y_train = []

    if TRAIN:
        print('Training our universal approximator')
    if LOAD:
        print('Loading pre-trained network')
    for i in range(100):
        for root, dirs, files in os.walk("./dataset/novi_dataset_pazljivo izdvajanje/batke/", topdown=True):
            for name in files:
                # print("Name = ", root)
                if name == "data.txt":

                    f_data = open(os.path.join(root, "data.txt"), "r")

                    for i, line in enumerate(f_data):
                        line = line.strip('\n')
                        line = re.split(r'\t+', line.rstrip('\t'))
                        if i == 1:
                            new_element = list(np.array(line).astype(np.float))[1:-1]
                            if REMOVE_MEASURING_POINTS:
                                #new_element.pop(5)
                                #new_element.pop(3)
                                #new_element.pop(2)
                                new_element.pop(0)

                            x_train.append(new_element)
                            if len(x_train[-1]) > 9:
                                print(root)
                                exit()
                        elif i == 3:
                            y_train.append(list(np.array(line).astype(np.float)))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.models import Sequential, save_model, load_model

    if LOAD:
        model = load_model("./istrenirani_modeli/istreniran_model_6/")
        print("Loaded old network")
    else:
        print("Created new network")
        model = Sequential([
            #Dense(576, input_shape=x_train.shape[1:]),
            #Activation('sigmoid'),
            #Dense(288),
            #Activation('sigmoid'),
            #Dense(144, input_shape=x_train.shape[1:]),
            #Activation('sigmoid'),
            Dense(72, input_shape=x_train.shape[1:]),
            Activation('sigmoid'),
            Dense(36),
            Activation('sigmoid')
        ])

    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])
    if TRAIN:
        for i in range(int(EPOCHS/100)):
            print("Trening "+str(i)+" od "+str(int(EPOCHS/100)))
            model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=BATCH)
            model.save('./istrenirani_modeli/istreniran_model_'+str(i))

            scores = model.evaluate(x_train, y_train, verbose=0)
            print(str(i) + " ->   Baseline Error: %.5f%%" % (100 - scores[1] * 100))

    print('Calculating result')

    x_train = []
    y_train = []
    broj_grafika = 0
    for root, dirs, files in os.walk("./dataset/novi_dataset_pazljivo izdvajanje/test/", topdown=True):
        for name in files:
            # print("Name = ", root)
            if name == "data.txt":

                f_data = open(os.path.join(root, "data.txt"), "r")
                mreza = None
                izlaz = None

                for i, line in enumerate(f_data):
                    line = line.strip('\n')
                    line = re.split(r'\t+', line.rstrip('\t'))
                    if i == 1:
                        print(root)
                        new_element = list(np.array(line).astype(np.float))[1:-1]
                        if REMOVE_MEASURING_POINTS:
                            #new_element.pop(5)
                            #new_element.pop(3)
                            #new_element.pop(2)
                            new_element.pop(0)

                        x_input = [new_element]
                        #y_res = sess.run([y], feed_dict={
                        #    x: x_input
                        #})
                        y_res = model.predict([x_input])

                        print("Input: ", x_input)
                        print("y_out: ", y_res)
                        new_dir = os.path.join("./test_results/" + root[11:])
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)

                        f_data = open(os.path.join(new_dir, "data.txt"), "w")

                        f_data.writelines("neural_network_result\n")
                        tmp_str = ""
                        for number in y_res[0]:
                            tmp_str += str(number)
                            tmp_str += ","
                        f_data.writelines(tmp_str[0:-1])
                        mreza = np.squeeze(y_res)
                        from scipy.signal import savgol_filter

                        # yhat = savgol_filter(mreza, 21, 5)

                    elif i == 3:
                        izlaz = np.array(line).astype(np.float)
                        izlaz = np.squeeze(izlaz)

                        XYZ_ref = colour.sd_to_XYZ(get_ref_sd(izlaz), cmfs, sd_illuminant)
                        Lab_ref = colour.XYZ_to_Lab(XYZ_ref / 100, illuminant_xy)

                        XYZ_machinelearning = colour.sd_to_XYZ(get_mach_lrn_sd(mreza), cmfs, sd_illuminant)
                        Lab_machinelearning = colour.XYZ_to_Lab(XYZ_machinelearning / 100, illuminant_xy)

                        print("Lab Racunato - EyeOnePro =", Lab_ref)
                        print("Lab Racunato - MachineLearning    =", Lab_machinelearning)

                        de = colour.delta_E(Lab_ref, Lab_machinelearning, method='CIE 2000')
                        des.append(de)
                        print("deltaE ref VS Machine_Learning = ", de)

                        plt.plot(mreza)
                        plt.plot(izlaz)
                        if broj_grafika < 5:
                            #plt.show()
                            broj_grafika += 1

    print("Delta Es : ", des)
    des.sort(reverse=True)
    print("Sorted D Es: ", des)
    print("Average = ", np.average(des))
    print("Max = ", np.max(des))
    print("Min = ", np.min(des))


    # Finally we save the graph to check that it looks like what we wanted
    #saver.save(sess, result_folder + '/data.chkp')



