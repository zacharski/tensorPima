import tensorflow as tf
import numpy as np
import diabetes as db
tf.set_random_seed(0)

diabetes = db.Diabetes()
XX = tf.placeholder(tf.float32, [None, 8])   # inputs  8 factors glucose level, etc.
Y_ = tf.placeholder(tf.float32, [None, 1])   # the correct output (wherther diabetes)
W = tf.Variable(tf.zeros([8, 1]))            # set weights to zero
b = tf.Variable(tf.zeros([1]))               # biases the same

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
#cross_entropy = tf.abs(Y_ - Y) #* 500.
#cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 500  # normalized for batches of 100 images,                                                          # *10 because  "mean" included an unwanted division by 10
cross_entropy = tf.reduce_mean(tf.abs(Y_ - Y)) #* 500.
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(Y, Y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = diabetes.next_batch()

    a, c, bb = sess.run([accuracy, cross_entropy, b], feed_dict={XX: batch_X, Y_: batch_Y})
    print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
    #print()
    a, c, bb, w, y, y_, cor = sess.run([accuracy, cross_entropy, b, W, Y, Y_, correct_prediction], feed_dict={XX: diabetes.testX, Y_: diabetes.testY})
    print(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    #print(bb)
    print("W" + str(w[0]))
    if i % 10 == 0:
        print(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y})

for i in range(5):
    training_step(i)

