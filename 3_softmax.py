import tensorflow as tf
import numpy as np
x_data = [
[1,1,1,1,1,1,1,1],
[2,3,3,5,7,2,6,7],
[1,2,4,5,5,5,6,7],
]
y_data = [
[0,0,0,0,0,0,1,1], #A
[0,0,0,1,1,1,0,0], #B
[1,1,1,0,0,0,0,0], #C
]

x_data = np.transpose(x_data)
y_data = np.transpose(y_data)

X = tf.placeholder("float", [None,3])
Y = tf.placeholder("float", [None,3])

W = tf.Variable(tf.zeros([3,3]))

H = tf.nn.softmax(tf.matmul(X,W))

alpha = 0.1

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), reduction_indices=1)) #?

optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for step in range(2001):
  sess.run(optimizer, feed_dict={X:x_data,Y:y_data})
  if step % 100 == 0:
    c = sess.run(cost, feed_dict={X:x_data,Y:y_data})
    #print (c,sess.run(W))
def _solver(x1,x2):
  from math import exp 
  prob=sess.run(H,feed_dict={X:[[1,x1,x2]]})[0]
  prob=[exp(i) for i in prob]
  s = sum(prob)
  prob=[i/s for i in prob]
  return prob
solver = _solver

print (solver(7,7))
print (solver(6,3))
print (solver(1,1))

