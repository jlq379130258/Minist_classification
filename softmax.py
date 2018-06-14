import gzip
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf

# MNIST data is stored in binary format,
# and we transform them into numpy ndarray objects by the following two utility functions


print ("Start processing MNIST handwritten digits data...")

train_features = np.fromfile("mnist_train_data",dtype=np.uint8)
train_labels = np.fromfile("mnist_train_label",dtype=np.uint8)
test_features = np.fromfile("mnist_test_data",dtype=np.uint8)
test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)


train_features = train_features.reshape(60000,45,45)
train_features = train_features.astype(np.float32)
test_features = test_features.reshape(10000,45,45)
test_features = test_features.astype(np.float32)
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

train_features=train_features.flatten()
train_features=train_features.reshape(60000,45*45)
test_features=test_features.flatten()
test_features=test_features.reshape(10000,45*45)

train_x_minmax = train_features / 255.0
test_x_minmax = test_features / 255.0



# We evaluate the softmax regression model by sklearn first
eval_sklearn = True
if eval_sklearn:
    print ("Start evaluating softmax regression model by sklearn...")
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    reg.fit(train_x_minmax, train_labels)
    np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
    test_y_predict = reg.predict(test_x_minmax)
    print ("Accuracy of test set: %f" % accuracy_score(test_labels, test_y_predict))

eval_tensorflow = False
batch_gradient = True

if eval_tensorflow:
    print ("Start evaluating softmax regression model by tensorflow...")
    # reformat y into one-hot encoding style
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_labels)
    train_y_data_trans = lb.transform(train_labels)
    test_y_data_trans = lb.transform(test_labels)

    x = tf.placeholder(tf.float32, [None, 2025])
    W = tf.Variable(tf.zeros([2025, 10]))
    b = tf.Variable(tf.zeros([10]))
    V = tf.matmul(x, W) + b
    y = tf.nn.softmax(V)

    y_ = tf.placeholder(tf.float32, [None, 10])

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    if batch_gradient:
        for step in range(300):
            sess.run(train, feed_dict={x: train_x_minmax, y_: train_y_data_trans})
            if step % 10 == 0:
                print ("Batch Gradient Descent processing step %d" % step)
        print ("Finally we got the estimated results, take such a long time...")
    else:
        for step in range(1000):
            sample_index = np.random.choice(train_x_minmax.shape[0], 100)
            batch_xs = train_x_minmax[sample_index, :]
            batch_ys = train_y_data_trans[sample_index, :]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
            if step % 100 == 0:
                print ("Stochastic Gradient Descent processing step %d" % step)
    np.savetxt('coef_softmax_tf.txt', np.transpose(sess.run(W)), fmt='%.6f')  # Save coefficients to a text file
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: test_x_minmax, y_: test_y_data_trans}))
