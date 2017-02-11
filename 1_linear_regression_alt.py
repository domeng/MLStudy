import tensorflow as tf
import sys

stepCount = 2001
showFreq = 20
try:
  stepCount = int(sys.argv[1])
  showFreq = int(sys.argv[2])
except:
  pass

x_data = [1,2,3,10,100]
y_data = [2,4,6,20,200]

W = tf.Variable(tf.random_uniform([1],-1.0,2.0))
b = tf.Variable(tf.random_uniform([1],-1.0,2.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
hypothesis =  W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.000001) # learning rate (alpha)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for step in range(stepCount):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % showFreq == 0:
      c = sess.run(cost)
      print (c,sess.run(W),sess.run(b))
