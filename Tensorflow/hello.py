import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hello = tf.constant('hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))