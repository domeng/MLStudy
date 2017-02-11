import tensorflow as tf

def showFuncName(func):
  def wrapper(*args, **kwargs):
    print ('='*5 + 'Result of %s' % (func.__name__) + '=' * 5)
    func(*args, **kwargs)
  return wrapper

@showFuncName
def tuto_helloWorld():
  hello = tf.constant('Hello, tensorflow!')
  sess = tf.Session()
  print (sess.run(hello).decode())

@showFuncName
def tuto_operation():
  a = tf.constant(2)
  b = tf.constant(3)
  c = a + b 
  with tf.Session() as sess:
    print (sess.run(c))

@showFuncName
def tuto_placeHolder():
  a = tf.placeholder(tf.int16)
  b = tf.placeholder(tf.int16)

  add = tf.add(a,b)
  mul = tf.mul(a,b)

  with tf.Session() as sess:
    print ("Add -> %i" % sess.run(add, feed_dict={a:2,b:3}))
    print ("Mul -> %i" % sess.run(mul, feed_dict={a:2,b:3}))

tuto_helloWorld()
tuto_operation()
tuto_placeHolder()

