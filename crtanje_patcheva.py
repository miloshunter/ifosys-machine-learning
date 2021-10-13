import time, os, argparse, io
# Tensorflow and numpy!
import tensorflow.keras.models
from tensorflow.keras import optimizers
import numpy as np
import statistics
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
diode_measurement_points = [400, 457, 517, 572, 632, 700]
#cp.plot_multi_sds(cmfs)

illuminant_xy=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]

svi_preko_3_imena = []
svi_preko_3_des = []


EPOCHS = 35
BATCH = 250
TRAIN_OR_LOAD = "LOAD"
LOAD_NUM = 71
LOAD_NAME = "./istrenirani_modeli/sa_kaggle/3_diode_kaggle/istreniran_model_"+str(LOAD_NUM)+"/"
number_of_diodes = 6

removed_samples = [
    (1, 27), (1, 31), (1, 32), (1, 35),(1, 37),(1, 38),(1, 41),(1, 43),(1, 44),(1, 45),
    (2, 35),(2, 42),(2, 43),(3, 42),(5, 43),(1, 42),(14, 39),(17, 37),
    (27, 28),(27, 38)
]


LOAD_NAMES = ["./istrenirani_modeli/konacno_rad/6dioda_1.15_0.96_3.12/",
              "./istrenirani_modeli/konacno_rad/5dioda_1.13_0.99_3.44/",
              "./istrenirani_modeli/konacno_rad/4diode_1.21_1.03_3.62/",
              "./istrenirani_modeli/konacno_rad/3diode_1.23_1.01_4.25/"
              ]

if number_of_diodes==6:
    LOAD_NAME = LOAD_NAMES[0]
if number_of_diodes==5:
    LOAD_NAME = LOAD_NAMES[1]
if number_of_diodes==4:
    LOAD_NAME = LOAD_NAMES[2]
if number_of_diodes==3:
    LOAD_NAME = LOAD_NAMES[3]

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

if number_of_diodes < 6:
    REMOVE_MEASURING_POINTS = True
else:
    REMOVE_MEASURING_POINTS = FalsePLOT = False

PLOT = True
PLOT_HIST = False
directory = os.path.dirname(os.path.realpath(__file__))

des = []

cell_width = 5
cell_height = 22
swatch_width = 48
margin = 12
topmargin = 40

ax = plt.subplot()
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)

def cm_to_inch(value):
    return value/2.54


#plt.rcParams['figure.figsize'] = [cm_to_inch(18), cm_to_inch(18)]
ax.set_aspect((42-5)/((168-2)/2))
num_x = 50
num_y = 35

ax.set_xlim(cell_width/2, cell_width/2 * num_x)
ax.set_ylim(cell_width*num_y, cell_width/2.)

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


def cm_to_inch(value):
    return value/2.54


plt.rcParams['figure.figsize'] = [cm_to_inch(18), cm_to_inch(6)]


if __name__ == '__main__':  # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument

    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.models import Sequential, save_model, load_model

    model = load_model(LOAD_NAME)

    print('Calculating result')

    x_train = []
    y_train = []

    test_names = []
    test_rows = []
    test_columns = []

    for row in range(1, 34):
        for columns in range(1, 46):
            test_rows.append(row)
            test_columns.append(columns)
            test_path = "./dataset/za_rad/sve_zajedno/red {}/{}/".format(str(row), str(columns))
            test_names.append(test_path)

    new_row = 1
    is_test = False
    i_x = 1
    i_y = 1
    for row, column, test_name in zip(test_rows, test_columns, test_names):
        is_test = False
        if row != new_row:
            new_row = row
            i_y = 1
            i_x = row
            pass
        if os.path.isdir(test_name):
            if os.path.isdir("./dataset/za_rad/test/red {}/{}/".format(str(row), str(column))):
                is_test = True
            name = os.listdir(test_name)[0]
            # print("Name = ", root)
            if name == "data.txt":
                root = test_name

                f_data = open(os.path.join(root, "data.txt"), "r")
                #f_data = open("./dataset/za_rad/test/red 27/5/data.txt")
                #PLOT=True
                #f_data = open("./dataset/za_rad/sve_zajedno/red 2/40/data.txt")
                mreza = None
                izlaz = None

                for i, line in enumerate(f_data):
                    line = line.strip('\n')
                    line = re.split(r'\t+', line.rstrip('\t'))
                    if i == 1:
                        print(root)
                        new_element = list(np.array(line).astype(np.float))[1:-1]
                        if REMOVE_MEASURING_POINTS:
                            diode_measurement_points = [400, 457, 517, 572, 632, 700]
                            if number_of_diodes == 5:
                                new_element.pop(0)
                                diode_measurement_points.pop(0)
                            elif number_of_diodes == 4:
                                new_element.pop(5)
                                new_element.pop(0)
                                diode_measurement_points.pop(5)
                                diode_measurement_points.pop(0)

                            elif number_of_diodes == 3:
                                new_element.pop(5)
                                new_element.pop(3)
                                new_element.pop(0)
                                diode_measurement_points.pop(5)
                                diode_measurement_points.pop(3)
                                diode_measurement_points.pop(0)

                            elif number_of_diodes == 2:
                                new_element.pop(5)
                                new_element.pop(4)
                                new_element.pop(2)
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
                        xyz = XYZColor(*[component / 100 for component in XYZ_ref])
                        rgb = convert_color(xyz, sRGBColor)
                        rgb_list = [1 * color for color in rgb.get_value_tuple()]
                        RGB_ref = rgb_list
                        RGB_ref = np.clip(RGB_ref, 0, 1)
                        XYZ_machinelearning = colour.sd_to_XYZ(get_mach_lrn_sd(mreza), cmfs, sd_illuminant)
                        Lab_machinelearning = colour.XYZ_to_Lab(XYZ_machinelearning / 100, illuminant_xy)

                        print("Lab Racunato - EyeOnePro =", Lab_ref)
                        print("Lab Racunato - MachineLearning    =", Lab_machinelearning)

                        de = colour.delta_E(Lab_ref, Lab_machinelearning, method='CIE 2000')
                        des.append(de)

                        if de > 3:
                            svi_preko_3_imena.append(root)
                            svi_preko_3_des.append(de)

                        print("deltaE ref VS Machine_Learning = ", de)
                        if PLOT: #Should i PLOT
                            i_x = row
                            i_y += 1


                            col_patch = matplotlib.patches.Rectangle(
                                xy=( cell_width/2*(i_y-1)+cell_width/2, cell_width*(i_x-1) + cell_width),
                                width=cell_width-3, height=cell_width-1,
                            )


                            ax.add_patch(col_patch)


                            def hex_to_rgb(hex):
                                hex = hex.lstrip('#')
                                hlen = len(hex)
                                return np.array([int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3)])


                            def gradient(hex1, hex2):
                                np1 = hex_to_rgb(hex1)
                                np2 = hex_to_rgb(hex2)
                                return np.linspace(np1[:, None], np2[:, None], 500, dtype=int)


                            if not is_test:
                                RGB_ref = (0.8, 0.8, 0.8)
                            else:
                                color = int(de / 5 * 500)
                                lista_boja = np.array(gradient('#00ff0a', '#ff0000'))
                                RGB_ref = (lista_boja[color][0][0] / 255, lista_boja[color][1][0] / 255,
                                           lista_boja[color][2][0] / 255)
                            col_patch.set_color(RGB_ref)


    #print("Delta Es : ", des)
    #des.sort(reverse=True)
    #print("Sorted D Es: ", des)
    #print("Average = ", np.average(des))
    #print("Median = ", statistics.median(des))
    #print("Max = ", np.max(des))
    #print("Min = ", np.min(des))

    plt.title("Map of ΔE₀₀ for "+str(number_of_diodes)+" diodes")
    #plt.title("Map of test samples")
    plt.savefig('/home/milos/maps{}.png'.format(str(number_of_diodes)), bbox_inches='tight')
    plt.show()

    #for ime, de in zip(svi_preko_3_imena, svi_preko_3_des):
    #    print("Des ", de, " ime ", ime[26:])

    # Finally we save the graph to check that it looks like what we wanted
    #saver.save(sess, result_folder + '/data.chkp')


