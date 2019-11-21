'''
# -*- coding:utf8 -*-
# Auther    : Mark
# File      :
# Date      :
# Des       :
    tf命令
'''

import tensorflow as tf

''' placeholder '''
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.add(a, b)
print('tf.add',end=' ')    #赋值
print(tf.Session().run(c,feed_dict={a:10,b:30}))


exp_a = tf.placeholder(tf.float32)
print('tf.exp ',end='')
print(tf.Session().run(tf.exp(exp_a),feed_dict={exp_a:2}))
