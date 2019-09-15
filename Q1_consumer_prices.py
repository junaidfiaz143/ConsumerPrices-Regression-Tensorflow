import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = np.genfromtxt("Q1-consumer-prices/train.csv", delimiter=',')

x_train = []
y_train = []

for i in range(len(data)):
    if i is not 0:
        x_train.append(data[i][0])
        y_train.append(data[i][1])

x_train = np.array(x_train)
y_train = np.array(y_train)

learning_rate = 0.0001
training_epochs = 500

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weights")


def model(X, w):
    return tf.multiply(X, w)


y_model = model(X, w)
loss = tf.reduce_mean(tf.square(Y - y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# """check accuracy"""
# correct_prediction = tf.equal(tf.to_float(tf.greater(y_model, 1.0)), Y)
# accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

# my_acc = tf.reduce_mean(tf.cast(tf.subtract(Y, y_model), dtype="float"))

sess = tf.Session()
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
sess.run(init)
sess.run(init_local)

for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})
        # print(sess.run(my_acc, feed_dict={X: x, Y: y}))

w_val = sess.run(w)
y_learned = x_train * w_val

plt.scatter(x_train, y_train, label="Data points")
plt.plot(x_train, y_learned, "r", label="Best fit line")
plt.legend()
plt.show()
