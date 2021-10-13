import time, os, argparse, io
# Tensorflow and numpy!
import tensorflow.keras.models
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, save_model, load_model

import numpy as np
import statistics
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colour
import colour
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
import colour.plotting as cp
import xlrd
import scipy.interpolate as ip
from numpy import diff

cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
#print(cmfs)
illuminant='D65'
sd_illuminant = colour.ILLUMINANTS_SDS[illuminant]
diode_measurement_points = []
#cp.plot_multi_sds(cmfs)

illuminant_xy=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]

EPOCHS = 35
BATCH = 250
TRAIN_OR_LOAD = "LOAD"
LOAD_NUM = 71
LOAD_NAME = "./istrenirani_modeli/sa_kaggle/3_diode_kaggle/istreniran_model_"+str(LOAD_NUM)+"/"
LOAD_NAMES = ["./istrenirani_modeli/konacno_rad/6dioda_1.15_0.96_3.12/",
              "./istrenirani_modeli/konacno_rad/5dioda_1.13_0.99_3.44/",
              "./istrenirani_modeli/konacno_rad/4diode_1.21_1.03_3.62/",
              "./istrenirani_modeli/konacno_rad/3diode_1.23_1.01_4.25/"
              ]

def cm_to_inch(value):
    return value/2.54


plt.rcParams['figure.figsize'] = [cm_to_inch(18), cm_to_inch(10)]


TITLE = "Estimated spectrums for test sample row 10 column 5"
f_data = open("./dataset/za_rad/test/red 10/5/data.txt")

number_of_diodes = 3
if number_of_diodes < 6:
    REMOVE_MEASURING_POINTS = True
else:
    REMOVE_MEASURING_POINTS = FalsePLOT = False

PLOT = True
PLOT_HIST = False
directory = os.path.dirname(os.path.realpath(__file__))

des = []

plot_symbols = [4, 5, 6, 7]
plot_colors = ['b', 'r', 'g', 'c']
plot_lines_styles = ['-.', '--', '-.', '--']
plot_labels = ['6 - ',
               '5 - ',
               '4 - ',
               '3 - ']


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

    ax = plt.subplot()
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Spectral Distribution [arb. dim.]")
    plt.title(TITLE)

    mreza = None
    izlaz = None
    new_element = []

    for i, line in enumerate(f_data):
        line = line.strip('\n')
        line = re.split(r'\t+', line.rstrip('\t'))
        if i == 1:
            for _ in range(4):
                new_element.append(list(np.array(line).astype(np.float))[1:-1])
                diode_measurement_points.append([400, 457, 517, 572, 632, 700])

            y_res = []
            for j in range(len(LOAD_NAMES)):
                if j == 1:
                    new_element[j].pop(0)
                    diode_measurement_points[j].pop(0)
                elif j == 2:
                    new_element[j].pop(5)
                    new_element[j].pop(0)
                    diode_measurement_points[j].pop(5)
                    diode_measurement_points[j].pop(0)

                elif j == 3:
                    new_element[j].pop(5)
                    new_element[j].pop(3)
                    new_element[j].pop(0)
                    diode_measurement_points[j].pop(5)
                    diode_measurement_points[j].pop(3)
                    diode_measurement_points[j].pop(0)

                model = load_model(LOAD_NAMES[j])
                model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mean_squared_error'])
                x_input = [new_element[j]]

                y_res.append(model.predict(x_input))

                print("Input: ", x_input)
                print("y_out: ", y_res)

        elif i == 3:
            izlaz = np.array(line).astype(np.float)
            izlaz = np.squeeze(izlaz)

            XYZ_ref = colour.sd_to_XYZ(get_ref_sd(izlaz), cmfs, sd_illuminant)
            Lab_ref = colour.XYZ_to_Lab(XYZ_ref / 100, illuminant_xy)
            xyz = XYZColor(*[component / 100 for component in XYZ_ref])
            rgb = convert_color(xyz, sRGBColor)
            rgb_list = [1 * color for color in rgb.get_value_tuple()]
            RGB_ref = rgb_list
            RGB_ref = np.clip(RGB_ref, 0, 1)

            for j in range(len(LOAD_NAMES)):

                mreza = y_res[j][0]

                XYZ_machinelearning = colour.sd_to_XYZ(get_mach_lrn_sd(mreza), cmfs, sd_illuminant)
                Lab_machinelearning = colour.XYZ_to_Lab(XYZ_machinelearning / 100, illuminant_xy)

                print("Lab Racunato - EyeOnePro =", Lab_ref)
                print("Lab Racunato - MachineLearning    =", Lab_machinelearning)

                de = colour.delta_E(Lab_ref, Lab_machinelearning, method='CIE 2000')
                des.append(de)
                print("deltaE ref VS Machine_Learning = ", de)
                if PLOT: #Should i PLOT

                    nm = np.asarray(np.linspace(380, 730, 36))  # example new x-axis

                    plt.plot(nm, mreza, linestyle=plot_lines_styles[j], c=plot_colors[j], label=plot_labels[j]+" ΔE₀₀: {:.2f}".format(de), zorder=1)

    for j in range(4):
        plt.plot(diode_measurement_points[j],
                 new_element[j], linestyle="None", c=plot_colors[j], marker=plot_symbols[j], markersize=7, zorder=10)

    plt.plot(nm, izlaz, c='0.6', label='Reference')
    col_patch = mpatches.Patch(label='Color patch')
    col_patch.set_color(RGB_ref)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(col_patch)
    plt.legend(handles=handles)
    plt.show()

    print("Delta Es : ", des)
    des.sort(reverse=True)
    print("Sorted D Es: ", des)
    print("Average = ", np.average(des))
    print("Median = ", statistics.median(des))
    print("Max = ", np.max(des))
    print("Min = ", np.min(des))

    if PLOT_HIST:
        plt.title("Histogram of ΔE00 for "+str(number_of_diodes)+" diodes")
        plt.xlabel("ΔE00 (CIE2000) metric")
        plt.ylabel("Number of test samples")
        plt.hist(des, range=(0, 5), bins=100)
        plt.show()



    # Finally we save the graph to check that it looks like what we wanted
    #saver.save(sess, result_folder + '/data.chkp')


