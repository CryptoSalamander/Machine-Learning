import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a = 2,b = 3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))