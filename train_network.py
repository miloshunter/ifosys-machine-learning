import time, os, argparse, io
# Tensorflow and numpy!
import tensorflow as tf
import numpy as np
import re

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
import matplotlib.pyplot as plt


directory = os.path.dirname(os.path.realpath(__file__))

batch_size = 100


# Our UA function
def univAprox(x, hidden_dim1=240, hidden_dim2=120, hidden_dim3=60):
    # The simple case is f: R -> R
    input_dim = 6
    output_dim = 36
    keep_prob = 0.8

    with tf.variable_scope('UniversalApproximator'):
        ua1_w = tf.get_variable(
            name='ua1_w'
            , shape=[input_dim, hidden_dim1]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        ua1_b = tf.get_variable(
            name='ua1_b'
            , shape=[hidden_dim1]
            , initializer=tf.constant_initializer(0.)
        )
        ua2_w = tf.get_variable(
            name='ua2_w'
            , shape=[hidden_dim1, hidden_dim2]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        ua2_b = tf.get_variable(
            name='ua2_b'
            , shape=[hidden_dim2]
            , initializer=tf.constant_initializer(0.)
        )
        ua3_w = tf.get_variable(
            name='ua3_w'
            , shape=[hidden_dim2, hidden_dim3]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        ua3_b = tf.get_variable(
            name='ua3_b'
            , shape=[hidden_dim3]
            , initializer=tf.constant_initializer(0.)
        )
        z1 = tf.matmul(x, ua1_w) + ua1_b
        dropped1 = tf.nn.dropout(z1, keep_prob=keep_prob)
        a1 = tf.nn.relu(dropped1)
        z2 = tf.matmul(a1, ua2_w) + ua2_b
        dropped2 = tf.nn.dropout(z2, keep_prob=keep_prob)
        a2 = tf.nn.relu(dropped2)
        z3 = tf.matmul(a2, ua3_w) + ua3_b
        dropped3 = tf.nn.dropout(z3, keep_prob=keep_prob)
        a3 = tf.nn.relu(dropped3)

        ua_v = tf.get_variable(
            name='ua_v'
            , shape=[hidden_dim3, output_dim]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        z = tf.matmul(a3, ua_v)

    return z


if __name__ == '__main__':  # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument

    # We build the computation graph
    with tf.variable_scope('Graph') as scope:
        # Our inputs will be a batch of values taken by our functions
        x = tf.placeholder(tf.float32, shape=[None, 6], name="x")
        y_true = tf.placeholder(tf.float32, [None, 36], name="y_true")

        # We define the ground truth and our approximation
        y = univAprox(x)

        # We define the resulting loss and graph it using tensorboard
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.square(y - y_true))
            # (Note the "_t" suffix here. It is pretty handy to avoid mixing
            # tensor summaries and their actual computed summaries)
            loss_summary_t = tf.summary.scalar('loss', loss)

            # We define our train operation using the Adam optimizer
        adam = tf.train.AdamOptimizer(learning_rate=0.003)
        train_op = adam.minimize(loss)

    # This is some tricks to push our matplotlib graph inside tensorboard
    with tf.variable_scope('TensorboardMatplotlibInput') as scope:
        # Matplotlib will give us the image as a string ...
        img_strbuf_plh = tf.placeholder(tf.string, shape=[])
        # ... encoded in the PNG format ...
        my_img = tf.image.decode_png(img_strbuf_plh, 4)
        # ... that we transform into an image summary
        img_summary = tf.summary.image(
            'matplotlib_graph'
            , tf.expand_dims(my_img, 0)
        )

        # We create a Saver as we want to save our UA after training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # We create a SummaryWriter to save data for TensorBoard
        result_folder = directory + '/results/' + str(int(time.time()))
        sw = tf.summary.FileWriter(result_folder, sess.graph)
        x_train = []
        y_train = []

        print('Training our universal approximator')
        sess.run(tf.global_variables_initializer())
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

            avg_cost = 0
            for i in range(int((len(x_train) - 1) / batch_size)):
                batch_x = x_train[i * batch_size:i * batch_size + batch_size]
                batch_y = y_train[i * batch_size:i * batch_size + batch_size]
            current_loss, loss_summary, _ = sess.run([loss, loss_summary_t, train_op],
                                                     feed_dict={x: batch_x, y_true: batch_y})

            # We leverage tensorboard by keeping track of the loss in real time
            sw.add_summary(loss_summary, i + 1)

            if (i + 1) % 1 == 0:
                print('batch: %d, loss: %f' % (i + 1, current_loss))

        print('Calculating result')

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
                            y_res = sess.run([y], feed_dict={
                                x: x_input
                            })
                            print("Input: ", x_input)
                            print("y_out: ", y_res)
                            new_dir = os.path.join("./test_results/" + root[11:])
                            if not os.path.exists(new_dir):
                                os.makedirs(new_dir)

                            f_data = open(os.path.join(new_dir, "data.txt"), "w")

                            f_data.writelines("neural_network_result")
                            tmp_str = ""
                            for number in y_res[0][0]:
                                tmp_str += str(number)
                                tmp_str += ","
                            f_data.writelines(tmp_str[0:-1])
                            mreza = np.squeeze(y_res)
                            from scipy.signal import savgol_filter

                            yhat = savgol_filter(mreza, 21, 5)

                        elif i == 3:
                            izlaz = np.array(line).astype(np.float)
                            izlaz = np.squeeze(izlaz)

                            plt.plot(yhat)
                            plt.plot(izlaz)
                            plt.show()

        # Finally we save the graph to check that it looks like what we wanted
        #saver.save(sess, result_folder + '/data.chkp')



