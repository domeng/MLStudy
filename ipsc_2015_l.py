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
    self.ans = '?'

  @staticmethod
  def open(path, ans=None):
    with open(path) as fr:
      lns = [[float(v)/255.0 for v in row.split()] for row in fr.readlines()]
    w = Word()
    w.ans = ans
    w.length = int(len(lns[0])/sz[0])
    for i in range(w.length):
      if ans is None:
        a = Alphabet(sz[0], sz[1], '?')
      else:
        a = Alphabet(sz[0], sz[1], ans[i]) 
      for y in range(sz[1]):
        for x in range(sz[0]):
          a.array[y][x][0] = lns[y][x+i*sz[0]]
      w.chars.append(a)
    return w

class Alphabet(object):
  def __init__(self,w,h,a):
    self.w = w
    self.h = h
    self.a = a
    self.array = np.zeros(shape=(h,w,1))
    if a in alp:
      self.output = np.zeros(shape=(len(alp)))
      self.output[alp.index(a)] = 1.0

  def shift(self, dx, dy):
    X = Alphabet(self.w, self.h, self.a)
    X.array = np.copy(self.array)
    for y in range(self.h):
      for x in range(self.w):
        if y + dy >= 0 and y + dy < self.h and x+dx>=0 and x+dx<self.w:
          X.array[y][x][0] = self.array[y+dy][x+dx][0]
        else:
          X.array[y][x][0] = 255
    return X

  def dirty(self, dirty_count):
    X = Alphabet(self.w, self.h, self.a)
    X.array = np.copy(self.array)
    for i in range(dirty_count):
      y = random.randint(0, self.h - 1)
      x = random.randint(0, self.w - 1)
      X.array[y][x][0] = 0
    return X

  def as_input_array(self):
    '''
    linear = np.zeros(shape=(1,self.w*self.h))
    for y in range(self.h):
      for x in range(self.w):
        linear[0][y*self.w + x] = self.array[y][x]
    return linear
    '''
    return self.array

  def as_output_array(self):
    '''
    linear = np.zeros(shape=(1,len(self.output)))
    for i in range(len(self.output)):
      linear[0][i] = self.output[i]
    return linear
    '''
    return self.output

  def image(self):
    img = Image.new('L', (self.w,self.h)) #w=100,h=70. usually.
    for y in range(self.h):
      for x in range(self.w):
        img.putpixel((x,y), (int(self.array[y][x][0] * 255),))
    return img

  def show(self):
    self.image().show()

  @staticmethod
  def open(path, alpha):
    with open(path) as fr:
      lns = [[[float(v)/255.0] for v in row.split()] for row in fr.readlines()]
    a = Alphabet(len(lns[0]), len(lns), alpha)
    a.array = np.array(lns)
    return a

  def shake(self):
    return self #.dirty(random.randint(800,1000))#.shift(random.randint(-2,2),random.randint(-3,3))#.dirty(random.randint(800,1000))

def load_alphabets():
  global alp
  dirs = os.path.join( '..','ipsc2016','l','alphabet')
  imgs = [Alphabet.open(os.path.join(dirs,a + '.in'), a) for a in alp]
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

def load_testword():
  with open(os.path.join( '..','ipsc2016','l','l1','sample.out'),'r') as fr:
    _ans = fr.readlines()
  return [Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % (idx+1)), _ans[idx]) for idx in range(190,200)]

def fetch_data(org, tw):
  dirties = []
  with open(os.path.join( '..','ipsc2016','l','l1','sample.out'),'r') as fr:
    _ans = fr.readlines()
  for idx in range(190):
    W = Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % (idx+1)), _ans[idx])
    for c in W.chars:
      dirties.append(c)
  random.shuffle(dirties)
  return dirties

if __name__=='__main__':
  org = load_alphabets()
  tw = load_testword()

  K = len(org)

  X = tf.placeholder('float', [None, 70, 100, 1])
  Y = tf.placeholder('float', [None, K])

  def init_weights(shape):  
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

  W = init_weights([3,3,1,32]) 
  W2 =init_weights([3,3,32,64])
  W3 =init_weights([3,3,64,128])

  keep_prob = tf.placeholder('float')

  L1A = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME'))
  L1B = tf.nn.max_pool(L1A, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  L1 = tf.nn.dropout(L1B, keep_prob)

  L2A = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME'))
  L2B = tf.nn.max_pool(L2A, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  L2 = tf.nn.dropout(L2B, keep_prob)

  W4 = init_weights([128*13*9, 625])
  WO = init_weights([625, K])

  L3A = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME'))
  L3B = tf.nn.max_pool(L3A, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  L3C = tf.reshape(L3B, [-1, W4.get_shape().as_list()[0]])
  L3 = tf.nn.dropout(L3C, keep_prob)

  #print(L3.get_shape().as_list()) = [None,13,9,128]

  keep_prob_hidden = tf.placeholder('float')

  L4A = tf.nn.relu(tf.matmul(L3,W4))
  L4 = tf.nn.dropout(L4A, keep_prob_hidden)

  H = tf.matmul(L4, WO)
  P = tf.argmax(H, 1)

  C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
  opt = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(C)

  init = tf.global_variables_initializer()

  dirties = fetch_data(org,tw)
  XS = np.array([c.as_input_array() for c in dirties])
  YS = np.array([c.as_output_array() for c in dirties])

  batch_size = 100
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
      cost = 0.
      for i in range(0,len(dirties),batch_size):
        j = min(len(dirties),i+batch_size)
        sess.run(opt, feed_dict={X: XS[i:j], Y: YS[i:j], keep_prob:0.8, keep_prob_hidden:0.5 })

      if epoch % 50 == 0:
        test = 0.
        cnt = 0
        pred = []
        for W in tw:
          tch = ""
          for c in W.chars:
            _c, _p = sess.run([C,P], feed_dict={X:[c.shake().as_input_array()],Y:[c.shake().as_output_array()], keep_prob:1.0, keep_prob_hidden:1.0})
            cnt += 1
            tch += "%s" % (alp[_p[0]], )
            test += _c
          pred.append(tch + "/" + W.ans)
        print ("epoch=%d , cost=%f test=%f [%s]" % (epoch, cost, test, "/".join(pred)))

    with open(os.path.join( '..','ipsc2016','l','l1','sample.out'),'r') as fr:
      _out = [x.strip() for x in fr.readlines()]
    for i in range(200,800):
      W = Word.open(os.path.join( '..','ipsc2016','l','l1','%03d.in' % (i+1)))
      _out[i] = ''
      for j in range(6):
        _p = sess.run(P, feed_dict={X:[W.chars[j].as_input_array()], keep_prob:1.0, keep_prob_hidden:1.0})
        _out[i] += alp[_p[0]]
    with open(os.path.join('l1.out'),'w') as fw:
      fw.write('\n'.join(_out))

