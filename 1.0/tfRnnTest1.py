'''
# -*- coding:utf8 -*-
# Auther    : Mark
# File      :
# Date      :
# Des       :
    工程介绍:http://blog.csdn.net/jerr__y/article/details/61195257
    TensorFlow命令介绍:https://wenku.baidu.com/view/7b416358a9114431b90d6c85ec3a87c240288ac1.html

'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

#设置GPU按需增长
def configGPU():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#导入数据
def importData():
    print('start import data')
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    print(mnist.train.images.shape)

configGPU()
importData()
#设置超参数
lr = 1e-3
#在训练和测试时,使用不同的batch_size 所以采用占位符方式
#batch_size = tf.placeholder(tf.int32)   #类型为tf.int32
'''
1.0版本后使用:
'''
keep_prod   = tf.placeholder(tf.float32,[])
batch_size  = tf.placeholder(tf.float32,[])

#每个时刻的输入特征为28维,即美食客输入一行,每行28个像素
input_size = 28

#时序持续长度为28,即没做一次预测,需要先输入28行
timestep_size = 28

#每个隐藏层节点数
hidden_size = 256

#LSTM layer 层数
layer_num = 2

#最后输出分类类别数量,如果是回归预测的话应该是1
class_num = 10

_x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,class_num])

'''             RNN过程              '''
#把784个点的字符信息还原成28*28的图片
#步骤1: RNN的输入shape = (batch_size,timestep_size,input_size)
x = tf.reshape(_x, [-1,28,28])

#步骤2: 定义一层LSTM_cell ,只需要说明hidden_size,即自动匹配输入的x的纬度
lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)

#步骤3: 添加dropout layer ,一般只设置output_keep_prob
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prod)

#步骤4: 调用MultiRNNCell来实现多层LSTM
mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num,state_is_tuple=True)

#步骤5: 用全0初始化state
init_state = mlstm_cell.zero_state(batch_size , dtype=tf.float32)

'''步骤6: 运行网络
#方法1,调用dynamic_rnn() 来运行构建好的网络
当 time_major == Falsee 时,outputs.shape = [batch_size ,timestep_size,hidden_size]
所以,可以取 h_state = state[-1][1]作为最后输出
state.shape = [layer_num,2,batch_size,hidden_size],
或者,可以取 h_state = state[-1][1]作为最后输出
最后输出纬度是[batch_size,hidden_size]
outputs,state = tf.nn.dynamic_rnn(mlstm_cell,inputs=x,initial=_state = init_state,time_major=False)
h_state= outputs[:,-1,:]#或 h_state = state[-1][1]


***********************  为理解lstm工作原理,上部函数通过自定义再次实现 ******************
(此处拷贝了作者的原文,为方便理解)通过查看文档会发现,RNNCell都提供了一个__call__()函数,我们可以用它来展开lstm按时间步迭代
方法2:按时间步展开计算
'''
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        #这里的state保存了每一层 lstm 的状态
        (cell_output,state) = mlstm_cell(x[:,timestep,:], state)
        outputs.append(cell_output)
h_state = outputs[-1]
print(h_state)

# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print "Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy)
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

# 计算测试数据的准确率
print "test accuracy %g"% sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]})




print('end')
