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

batch_size = 50


# Our UA function
def univAprox(x, hidden_dim=50):
    # The simple case is f: R -> R
    input_dim = 6
    output_dim = 36

    with tf.variable_scope('UniversalApproximator'):
        ua_w = tf.get_variable(
            name='ua_w'
            , shape=[input_dim, hidden_dim]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        ua_b = tf.get_variable(
            name='ua_b'
            , shape=[hidden_dim]
            , initializer=tf.constant_initializer(0.)
        )
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z)  # we now have our hidden_dim activations

        ua_v = tf.get_variable(
            name='ua_v'
            , shape=[hidden_dim, output_dim]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        z = tf.matmul(a, ua_v)

    return z


if __name__ == '__main__':  # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_neurons", default=250, type=int, help="Number of neurons or the UA")
    args = parser.parse_args()

    # We build the computation graph
    with tf.variable_scope('Graph') as scope:
        # Our inputs will be a batch of values taken by our functions
        x = tf.placeholder(tf.float32, shape=[None, 6], name="x")
        y_true = tf.placeholder(tf.float32, [None, 36], name="y_true")

        # We define the ground truth and our approximation
        y = univAprox(x, args.nb_neurons)

        # We define the resulting loss and graph it using tensorboard
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.square(y - y_true))
            # (Note the "_t" suffix here. It is pretty handy to avoid mixing
            # tensor summaries and their actual computed summaries)
            loss_summary_t = tf.summary.scalar('loss', loss)

            # We define our train operation using the Adam optimizer
        adam = tf.train.AdamOptimizer(learning_rate=1e-3)
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
        for i in range(200):
            for root, dirs, files in os.walk("./dataset/trening/", topdown=True):
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
        # We compute a dense enough graph of our functions
        x_input = [
                 [0.0676666666667,	0.167142857143,	0.0604744525547,	0.023125,	0.789285714286,	0.852142857143]
        ]
        y_res = sess.run([y], feed_dict={
            x: x_input
        })
        print("Input: ", x_input)
        print("y_out: ", y_res)
        data = np.squeeze(y_res)
        plt.plot(data)
        plt.show()

        # Finally we save the graph to check that it looks like what we wanted
        saver.save(sess, result_folder + '/data.chkp')


'''
# Python optimisation variables
learning_rate = 0.05
epochs = 10
batch_size = 100

# declare the training data placeholders
x = tf.placeholder(tf.float32, [None, 6])
# now declare the output data placeholder
y = tf.placeholder(tf.float32, [None, 36])

W1 = tf.Variable(tf.random_normal([6, 100], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([100]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([100, 36], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([36]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.add(tf.matmul(hidden_out, W2), b2)

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    for epoch in range(epochs):

        x_train = []
        y_train = []

        for root, dirs, files in os.walk("./dataset/trening/", topdown=True):
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
        for i in range(int((len(x_train)-1)/batch_size)):
            batch_x = x_train[i*batch_size:i*batch_size+batch_size]
            batch_y = y_train[i*batch_size:i*batch_size+batch_size]
            _, c = sess.run([optimiser, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            # print("Cost = ", c)
            avg_cost += c
        avg_cost = avg_cost/len(x_train)

        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    rezultat = sess.run(y_ , feed_dict={x: [[0.0883333333333, 0.172857142857, 0.108649635036, 0.076875, 0.795714285714, 0.852857142857]]})
    print("Rez: ", rezultat)
'''



