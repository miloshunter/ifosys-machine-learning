import time, os, argparse, io
# Tensorflow and numpy!
from tensorflow.keras import optimizers
import numpy as np
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt


directory = os.path.dirname(os.path.realpath(__file__))

batch_size = 100



if __name__ == '__main__':  # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument

    # We create a SummaryWriter to save data for TensorBoard
    result_folder = directory + '/results/' + str(int(time.time()))
    x_train = []
    y_train = []

    print('Training our universal approximator')
    for i in range(100):
        for root, dirs, files in os.walk("./dataset/sve/", topdown=True):
            for name in files:
                # print("Name = ", root)
                if name == "data.txt":

                    f_data = open(os.path.join(root, "data.txt"), "r")

                    for i, line in enumerate(f_data):
                        line = line.strip('\n')
                        line = re.split(r'\t+', line.rstrip('\t'))
                        if i == 1:
                            x_train.append(list(np.array(line).astype(np.float))[1:-1])
                            if len(x_train[-1]) > 9:
                                print(root)
                                exit()
                        elif i == 3:
                            y_train.append(list(np.array(line).astype(np.float)))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Dense(256, input_shape=x_train.shape[1:]),
        Activation('sigmoid'),
        Dense(128),
        Activation('sigmoid'),
        Dense(64),
        Activation('sigmoid'),
        Dense(36),
        Activation('sigmoid')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(x_train, y_train, epochs=500, batch_size=500)

    scores = model.evaluate(x_train, y_train, verbose=0)
    print("Baseline Error: %.5f%%" % (100 - scores[1] * 100))

    print('Calculating result')

    x_train = []
    y_train = []
    for root, dirs, files in os.walk("./dataset/test/", topdown=True):
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
                        niz = list(np.array(line).astype(np.float))[1:-1]
                        x_train.append(niz)
                        if len(x_train[-1]) > 9:
                            print(root)
                            exit()
                        x_input = [niz]
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

                        plt.plot(mreza)
                        plt.plot(izlaz)
                        plt.show()

    # Finally we save the graph to check that it looks like what we wanted
    #saver.save(sess, result_folder + '/data.chkp')



