# * Logistic Classfication?
# X = R^n , Y=[0 or 1]^n
# X 값에 따라 어떤 기준점 z를 기점으로 Y 값이 0에서 1로 바뀐다면
# 이를 선형회귀(H(x)=Wx+b)로 분석하기는 힘들다. (X_i >> z인 데이타가 있으면 H(x)>=0.5라는 기준으로 판단하기 힘든 기울기가 나옴.)

# > 따라서, 가설함수를 sigmoid function (or logistic function)으로 대체한다.
# H(x) = 1 / (1+exp(-Ax+B)), 0 if x->-inf, 1 if x->inf
# ! 가설함수는 바뀌었으나 Linear Regression때의 Cost Function을 그대로 사용하면 
# 무한한 local minima가 만들어질 수 있다. -> GradientOpt방법을 쓸수없다

# > 따라서, Cost function 역시 log를 이용하여 local minima를 없애는 방향으로 간다.
# C(H(x),Y) := -log(H(x)) if Y=1 , -log(1-H(x)) if Y = 0
# 하나의 수식으로 정리하면 C := -Ylog(H(x)) - (1-Y)log(1-H(x))
# 이렇게하면 아래로 볼록한 그래프가 나오므로 GradientOpt방법을 쓸수있다

import tensorflow as tf
import numpy as np

def oneVariableTest(x_data,y_data):
  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
  b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

  hypothesis = tf.div(1.,1.+tf.exp(-W*X+b))
  cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

  a = tf.Variable(0.1)
  optimizer = tf.train.GradientDescentOptimizer(a)
  train = optimizer.minimize(cost)

  init = tf.global_variables_initializer()

  sess = tf.Session()
  sess.run(init)
  for step in range(2001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % 500 == 0:
      c = sess.run(cost, feed_dict={X:x_data,Y:y_data})
      print (c,sess.run(W),sess.run(b))
  
  # return a function which will answer the question
  def answerFunc(x):
    v = sess.run(hypothesis,feed_dict={X:x})
    return v, 1 if v>=0.5 else 0
  return answerFunc

def twoVariableTest(x_data,y_data): #same but expand x to x' as 2D-vector [1,x] for merging b to W
  x_data = [x_data,[1]*len(x_data)] #WARN: form as not [[x1,1],[x2,1]...] but [[x1,x2,x3,..],[1,1,1...]]

  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0)) #changed
  h = tf.matmul(W,X)
  hypothesis = tf.div(1.,1.+tf.exp(-h)) #changed

  cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

  a = tf.Variable(0.1)
  optimizer = tf.train.GradientDescentOptimizer(a)
  train = optimizer.minimize(cost)

  init = tf.global_variables_initializer()

  sess = tf.Session()
  sess.run(init)
  for step in range(2001):
    sess.run(train, feed_dict={X:x_data,Y:y_data})
    if step % 500 == 0:
      c = sess.run(cost, feed_dict={X:x_data,Y:y_data})
      print (c,sess.run(W))

  # return a function which will answer the question
  def answerFunc(x):
    v = sess.run(hypothesis,feed_dict={X:[[x],[1]]})
    return v, 1 if v>=0.5 else 0
  return answerFunc


def dataGen(n,minV,maxV,threshold):
  import random
  x = [random.uniform(minV,maxV) for _ in range(n)] 
  y = [0. if x[i] < threshold else 1. for i in range(n)]
  return x,y

x_data,y_data = dataGen(100,0.0,1.0,0.66)
first_1 = min(u[0] for u in zip(x_data,y_data) if u[1]==1)
last_0 = max(u[0] for u in zip(x_data,y_data) if u[1]==0)
a1,a2 = oneVariableTest(x_data,y_data),twoVariableTest(x_data,y_data)
print ((last_0,first_1))
for i in range(100):
  a,b = a1(i/100.0)
  x,y = a2(i/100.0)
  chk = '?'
  if i/100.0 > first_1:
    chk = '1'
  elif i/100.0 < last_0:
    chk = '0'
  print ("%.3f [%f %f] vs [%f %f] [%s]" %(i/100.0,a,b,x,y,chk))
