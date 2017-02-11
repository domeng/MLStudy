import tensorflow as tf
import sys

stepCount = 2001
showFreq = 20
try:
  stepCount = int(sys.argv[1])
  showFreq = int(sys.argv[2])
except:
  pass

x_data = [1,2,3,]
y_data = [1,2,3,]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.05) # learning rate (alpha)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables() # deprecated?

with tf.Session() as sess:
  sess.run(init)

  for step in range(stepCount):
    sess.run(train)
    if step % showFreq == 0:
      print ((step, sess.run(cost), sess.run(W), sess.run(b)))
