#-*- encoding:utf-8 -*-
# assume we already have a data at "../ipsc2016/l" (alphabet,l1,l2)
# all of alphabet size = (100,70)

import os
import pdb
import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from PIL import Image

alp = [chr(x) for x in range(ord('a'),ord('z')+1) if not chr(x) in 'fhx']
sz = (100,70)
ans = ['mweaul','iopnje','doezws','qwpivl'] #1..4

class Word(object):
  def __init__(self):
    self.length = 0
    self.chars = []

  @staticmethod
  def open(path, ans=None):
    with open(path) as fr:
      lns = [[float(v)/255.0 for v in row.split()] for row in fr.readlines()]
    w = Word()
    w.length = int(len(lns[0])/sz[0])
    for i in range(w.length):
      if ans is None:
        a = Alphabet(sz[0], sz[1], '?')
      else:
        a = Alphabet(sz[0], sz[1], ans[i]) 
      for y in range(sz[1]):
        for x in range(sz[0]):
          a.array[y][x] = lns[y][x+i*sz[0]]
      w.chars.append(a)
    return w

class Alphabet(object):
  def __init__(self,w,h,a):
    self.w = w
    self.h = h
    self.a = a
    self.array = np.zeros(shape=(h,w))
    if a in alp:
      self.output = np.zeros(shape=(len(alp),))
      self.output[alp.index(a)] = 1.0

  def shift(self, dx, dy):
    X = Alphabet(self.w, self.h, self.a)
    X.array = np.copy(self.array)
    for y in range(self.h):
      for x in range(self.w):
        if y + dy >= 0 and y + dy < self.h and x+dx>=0 and x+dx<self.w:
          X.array[y][x] = self.array[y+dy][x+dx]
        else:
          X.array[y][x] = 255
    return X

  def dirty(self, dirty_count):
    X = Alphabet(self.w, self.h, self.a)
    X.array = np.copy(self.array)
    for i in range(dirty_count):
      y = random.randint(0, self.h - 1)
      x = random.randint(0, self.w - 1)
      X.array[y][x] = 0
    return X

  def as_input_array(self):
    linear = np.zeros(shape=(1,self.w*self.h))
    for y in range(self.h):
      for x in range(self.w):
        linear[0][y*self.w + x] = self.array[y][x]
    return linear

  def as_output_array(self):
    linear = np.zeros(shape=(1,len(self.output)))
    for i in range(len(self.output)):
      linear[0][i] = self.output[i]
    return linear

  def image(self):
    img = Image.new('L', (self.w,self.h)) #w=100,h=70. usually.
    for y in range(self.h):
      for x in range(self.w):
        img.putpixel((x,y), (int(self.array[y][x] * 255),))
    return img

  def show(self):
    self.image().show()

  @staticmethod
  def open(path, alpha):
    with open(path) as fr:
      lns = [[float(v)/255.0 for v in row.split()] for row in fr.readlines()]
    a = Alphabet(len(lns[0]), len(lns), alpha)
    a.array = np.array(lns)
    return a

def load_alphabets():
  global alp
  dirs = os.path.join( '..','ipsc2016','l','alphabet')
  imgs = [Alphabet.open(os.path.join(dirs,a + '.in'), a) for a in alp]
  #x = imgs[1].dirty(random.randint(500,800))
  #x.show()
  #pdb.set_trace()
  return imgs

def chars2d_to_image(data):
  W = len(data[0])
  H = len(data)
  img = Image.new('L', (W*sz[0],H*sz[1])) #w=100,h=70. usually.
  for y in range(H):
    for x in range(W):
      if data[y][x] is None:
        continue
      img.paste(data[y][x].image(),(x*sz[0],y*sz[1]))
  return img


if __name__=='__main__':
  org = load_alphabets()

  K = len(alp)

  X = tf.placeholder('float', [None, 7000])
  Y = tf.placeholder('float', [None, K])

  W1 = tf.Variable(tf.random_normal([7000, 256]))
  W2 = tf.Variable(tf.random_normal([256, 256]))
  W3 = tf.Variable(tf.random_normal([256, K]))

  B1 = tf.Variable(tf.random_normal([256]))
  B2 = tf.Variable(tf.random_normal([256]))
  B3 = tf.Variable(tf.random_normal([K]))

  L1 = tf.nn.relu(tf.add(tf.matmul(X,W1), B1))
  L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
  H = tf.add(tf.matmul(L2, W3), B3)

  C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
  opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(C)

  init = tf.global_variables_initializer()

  '''
  for i in range(4):
    W = Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % (i+1)))
    pic = [W.chars,[org[alp.index(x)].shift(random.randint(-22,22),random.randint(-6,6)).dirty(random.randint(800,1000)) for x in ans[i] ]]
    chars2d_to_image(pic).show()
  '''
  dirties = []
  for i in org:
    for j in range(4):
      x = i.shift(random.randint(-20,20),random.randint(-6,6)).dirty(random.randint(800,1000))
      dirties.append(x)
  random.shuffle(dirties)

  before_train = datetime.now()
  batch_size = 100
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
      cost = 0.
      for i in range(0,len(dirties),batch_size):
        _in = dirties[i].as_input_array()
        _out = dirties[i].as_output_array()
        for j in range(i+1,min(len(dirties),i+batch_size)):
          _in = np.concatenate((_in, dirties[j].as_input_array()))
          _out = np.concatenate((_out, dirties[j].as_output_array()))
        sess.run(opt, feed_dict={X: _in, Y: _out })
        cost += sess.run(C, feed_dict={X: _in, Y:_out })
      if epoch % 20 == 0:
        test = 0.
        cnt = 0
        for i in range(4):
          W = Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % (i+1)), ans[i])
          for c in W.chars:
            test += sess.run(C, feed_dict={X:c.as_input_array(),Y:c.as_output_array()})
            cnt += 1
        print ("epoch=%d , cost=%f test=%f" % (epoch, cost/len(dirties), test/cnt))

    after_train = datetime.now()
    train_time = after_train - before_train
    print("train_time : ", train_time.total_seconds())

'''
        for l in range(1,5):
          i = random.randint(1,800)
          W = Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % i))
          pic = [W.chars,[]]
          #for i in range(6):
          #  pic[0].append(dirties[random.randint(0,len(dirties)-1)])
          for i in pic[0]:
            x = sess.run(H, feed_dict={X: i.as_input_array()})
            prob = list(sess.run(tf.nn.softmax(x[0])))
            index = prob.index(max(prob))
            pic[1].append(org[index])
          chars2d_to_image(pic).show()
'''
