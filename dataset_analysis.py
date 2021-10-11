import time, os, argparse, io
# Tensorflow and numpy!
#import tensorflow.keras.models
#from tensorflow.keras import optimizers
import numpy as np
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colour
import colour
import colour.plotting as cp
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
import xlrd
import scipy.interpolate as ip
from numpy import diff

cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
#print(cmfs)
illuminant='D65'
sd_illuminant = colour.ILLUMINANTS_SDS[illuminant]

#cp.plot_multi_sds(cmfs)

illuminant_xy=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]

EPOCHS = 35
BATCH = 250
TRAIN_OR_LOAD = "LOAD"
LOAD_NUM = 87

RED = 7
KOLONA = 40

LOAD_NAME = "./istrenirani_modeli/sa_kaggle/6_dioda_bez_tamnih/istrenirani_modeli/istreniran_model_"+str(LOAD_NUM)+"/"
#LOAD_NAME = "./istrenirani_modeli/isteniran_model_99"
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
    #result_folder = directory + '/results/' + str(int(time.time()))
    #x_train = []
    #y_train = []
#
    #if TRAIN_OR_LOAD == "TRAIN":
    #    print('Training our universal approximator')
#
    #    for i in range(100):
    #        for root, dirs, files in os.walk("./dataset/za_rad/trening_dodatni_tamni/", topdown=True):
    #            for name in files:
    #                # print("Name = ", root)
    #                if name == "data.txt":
#
    #                    f_data = open(os.path.join(root, "data.txt"), "r")
#
    #                    for i, line in enumerate(f_data):
    #                        line = line.strip('\n')
    #                        line = re.split(r'\t+', line.rstrip('\t'))
    #                        if i == 1:
    #                            new_element = list(np.array(line).astype(np.float))[1:-1]
    #                            if REMOVE_MEASURING_POINTS:
    #                                new_element.pop(5)
    #                                new_element.pop(3)
    #                                new_element.pop(2)
    #                                new_element.pop(0)
#
    #                            x_train.append(new_element)
    #                            if len(x_train[-1]) > 9:
    #                                print(root)
    #                                exit()
    #                        elif i == 3:
    #                            y_train.append(list(np.array(line).astype(np.float)))
#
    #    x_train = np.asarray(x_train)
    #    y_train = np.asarray(y_train)
    #else:
    #    print('Loading pre-trained network')
#
    #from tensorflow.keras.layers import Dense, Activation
    #from tensorflow.keras.models import Sequential, save_model, load_model
#
    #if TRAIN_OR_LOAD == "LOAD":
    #    model = load_model(LOAD_NAME)
    #else:
    #    #model = Sequential([
    #    #    Dense(576),
    #    #    Activation('sigmoid'),
    #    #    Dense(288),
    #    #    Activation('sigmoid'),
    #    #    Dense(144),
    #    #    Activation('sigmoid'),
    #    #    Dense(72),
    #    #    Activation('sigmoid'),
    #    #    Dense(36),
    #    #    Activation('sigmoid')
    #    #])
    #    model = Sequential([
    #        # Dense(576),
    #        # Activation('sigmoid'),
    #        # Dense(288),
    #        # Activation('sigmoid'),
    #        # Dense(144, activation='relu'),
    #        Dense(72, activation='sigmoid', input_shape=x_train.shape[1:]),
    #        Dense(36, activation='sigmoid'),
    #    ])
#
    #    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])
    #if TRAIN_OR_LOAD == "TRAIN":
    #    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH)
    #    model.save('istreniran_model')
#
    #    scores = model.evaluate(x_train, y_train, verbose=0)
    #    print("Baseline Error: %.5f%%" % (100 - scores[1] * 100))
#
    #print('Calculating result')

    #x_train = []
    #y_train = []
    files = []
    plot_color = []
    plot_patches = []
    plot_labels = []
    plot_symbols = ['D', 'o', '*', 'X', 'v']

    TITLE = 'Saturated RGB and White Colors'

    files.append("./dataset/za_rad/sve_zajedno/red 1/27/")
    plot_labels.append('♦ R17-C37 - Bad measurement')
    plot_color.append('r')

    files.append("./dataset/za_rad/sve_zajedno/red 1/32/")
    plot_labels.append('• R18-C37 - Color for comparison')
    plot_color.append('b')

    # files.append("./dataset/za_rad/sve_zajedno/red 2/40/")
    # plot_labels.append('• R18-C36')
    # plot_color.append('y')
    #
    # files.append("./dataset/za_rad/sve_zajedno/red 6/39/")
    # plot_labels.append('★ R19-C37')
    # plot_color.append('b')
    #
    # files.append("./dataset/za_rad/sve_zajedno/red 3/39/")
    # plot_labels.append('✖ R16-C37')
    # plot_color.append('r')

    brojac = 0
    for name in files:
        # print("Name = ", root)
        f_data = open(os.path.join(name, "data.txt"), "r")
        mreza = None
        izlaz = None

        for i, line in enumerate(f_data):
            line = line.strip('\n')
            line = re.split(r'\t+', line.rstrip('\t'))
            if i == 1:
                print(name)
                new_element = list(np.array(line).astype(np.float))[1:-1]
                if REMOVE_MEASURING_POINTS:
                    new_element.pop(5)
                    new_element.pop(3)
                    new_element.pop(2)
                    new_element.pop(0)

                x_input = [new_element]
                #y_res = sess.run([y], feed_dict={
                #    x: x_input
                #})
                # y_res = model.predict([x_input])
                #
                # print("Input: ", x_input)
                # print("y_out: ", y_res)
                # new_dir = os.path.join("./test_results/" + root[11:])
                # if not os.path.exists(new_dir):
                #     os.makedirs(new_dir)
                #
                # f_data = open(os.path.join(new_dir, "data.txt"), "w")
                #
                # f_data.writelines("neural_network_result\n")
                # tmp_str = ""
                # for number in y_res[0]:
                #     tmp_str += str(number)
                #     tmp_str += ","
                # f_data.writelines(tmp_str[0:-1])
                # mreza = np.squeeze(y_res)
                from scipy.signal import savgol_filter

                # yhat = savgol_filter(mreza, 21, 5)

            elif i == 3:
                izlaz = np.array(line).astype(np.float)
                izlaz = np.squeeze(izlaz)

                XYZ_ref = colour.sd_to_XYZ(get_ref_sd(izlaz), cmfs, sd_illuminant)
                Lab_ref = colour.XYZ_to_Lab(XYZ_ref / 100, illuminant_xy)

                xyz = XYZColor(*[component/100 for component in XYZ_ref])
                rgb = convert_color(xyz, sRGBColor)
                rgb_list = [1 * color for color in rgb.get_value_tuple()]
                RGB_ref = rgb_list
                RGB_ref = np.clip(RGB_ref, 0, 1)

                #XYZ_machinelearning = colour.sd_to_XYZ(get_mach_lrn_sd(mreza), cmfs, sd_illuminant)
                #Lab_machinelearning = colour.XYZ_to_Lab(XYZ_machinelearning / 100, illuminant_xy)

                print("Lab Racunato - EyeOnePro =", Lab_ref)
                #print("Lab Racunato - MachineLearning    =", Lab_machinelearning)

                #de = colour.delta_E(Lab_ref, Lab_machinelearning, method='CIE 2000')
                #des.append(de)
                #print("deltaE ref VS Machine_Learning = ", de)



                nm = np.asarray(np.linspace(380, 730, 36))  # example new x-axis
                # print(nm)
                #   nm=np.linspace(380, 730, 36) # example new x-axis
                #    print(nm)
                #   sd=0
                plot_patches.append(mpatches.Patch(label=plot_labels[brojac]))
                plot_patches[brojac].set_color(RGB_ref)
                data_plot = dict(zip(nm, izlaz))
                plt.title(TITLE)
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Spectral Distribution [arb. dim.]")
                plt.plot(nm, izlaz, '--', c=str(plot_color[brojac]))
                plt.plot([400, 457, 517, 572, 632, 700],
                         new_element, plot_symbols[brojac], c=str(plot_color[brojac]))
                #cp.plot_single_sd(sd, standalone=False)
                #plt.plot(sd, '-.')
                brojac += 1
                #plt.plot(nm, mreza, '-.')
    plt.legend(handles=plot_patches)
    plt.show()

    #print("Delta Es : ", des)
    #des.sort(reverse=True)
    #print("Sorted D Es: ", des)
    #print("Average = ", np.average(des))
    #print("Max = ", np.max(des))
    #print("Min = ", np.min(des))



    # Finally we save the graph to check that it looks like what we wanted
    #saver.save(sess, result_folder + '/data.chkp')


