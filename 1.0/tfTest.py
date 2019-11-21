'''
# -*- coding:utf8 -*-
# Auther    : Mark
# File      :
# Date      :
# Des       :
    参照网址:http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
'''
#--------------------------------
def printTitle(fun):
    print ('__FUNC__ '+fun.__name__,end='  >>  ')
#--------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#构建图

#创建一个常量op,产生一个1*2的矩阵,这个op被加到默认图中
#构造器的返回值代表常量 op 的返回值
matrix1 = tf.constant([[3.,3.]])

#创建另外一个常量 op ,产生一个2*1的矩阵
matrix2 = tf.constant([[2.],[2.]])

#创建一个矩阵乘法 matmul op,吧'matrix1' 和 'matrix2'作为输入
'''返回值'product'代表 矩阵乘法 的结果'''
product = tf.matmul(matrix1, matrix2)

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
''' 返回值 'result' 是一个 numpy `ndarray` 对象.xxx
sess = tf.Session()         #显示操作,不会自动关闭
result = sess.run(product)
sess.close()
print(result)
'''
''''''
#通过with代码块可以自动实现sess关闭
with tf.Session() as sess:
    result = sess.run(product)
    # print('__FUNC_  tf.matmul(matrix1,matrix2) >>',end='   ')
    printTitle(tf.matmul)
    print(result)


#--------------------------------  使用GPU  --------------------------------
'''
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        gpu_matrix1 = tf.constant([[3.,3.]])
        gpu_matrix2 = tf.constant([[2.],[2.]])
        gpu_product = tf.matmul(gpu_matrix1, gpu_matrix2)
        print('gpu:',end='  ')
        sess.run(gpu_product)
'''

#--------------------------------  交互式使用  --------------------------------
'''
文档中的 Python 示例使用一个会话 Session 来 启动图, 并调用 Session.run() 方法执行操作.

为了便于使用诸如 IPython 之类的 Python 交互环境,
可以使用 InteractiveSession 代替 Session 类,
使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run().
这样可以避免使用一个变量来持有会话.

#该方法可能由于版本问题,sub方法无法使用
sess_sess = tf.InteractiveSession()
sess_x = tf.Variable([1.0,2.0])
sess_a = tf.constant([3.0,3.0])

#使用初始化器 initializer op 的run()方法初始化'sess_x'
sess_x.initializer.run()

#增加一个减法 sub op ,从 'sess_x'减去'sess_a',运行减法op,
sess_sub = tf.sub(sess_x,sess_a)
print('__FUNC__ tf.sub(x,a)',end='   >>  ')
print(sess_sub.eval())
'''
#--------------------------------  变量  --------------------------------
'''
Variables for more details. 变量维护图执行过程中的状态信息.
下面的例子演示了如何使用变量实现一个简单的计数器. 参见 变量 章节了解更多细节.
'''
#创建一个变量,初始化为标量 0
var_state = tf.Variable(0,name='counter')

#创建一个 op,起作用是使 state 增加 1
var_one = tf.constant(1)
var_value = tf.add(var_state, var_one)
var_update = tf.assign(var_state, var_value)

#启动图后, 变量必须经过'初始化',(init)op初始化,
#先增加一个'初始化'op到图中
var_init_op = tf.global_variables_initializer()

#启动图 运行 op
with tf.Session() as sess:
    #运行'init'op
    sess.run(var_init_op)
    #打印'state' 的初始化
    printTitle(tf.assign)
    print(sess.run(var_state))
    #运行 op,更新'state', 并打印'state'
    for _ in range(3):
        sess.run(var_update)
        printTitle(sess.run)
        print(sess.run(var_state))


#--------------------------------  取值(Fetch)  --------------------------------
'''
为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时,
传入一些 tensor, 这些 tensor 会帮助你取回结果.
在之前的例子里, 我们只取回了单个节点 state, 但是你也可以取回多个 tensor:
'''
fet_input1 = tf.constant(3.0)
fet_input2 = tf.constant(2.0)
fet_input3 = tf.constant(5.0)
fet_intermed = tf.add(fet_input1,fet_input2)
fet_mul = tf.multiply(fet_input3,fet_intermed)
with tf.Session() as sess:
    '''
    关于run的解释:
    sess.run(param,feed_dict)
    其中:@param为op向量
        @feed_dict 为变量替换值,仅在计算时替换值变量,不影响原变量值
    '''
    result = sess.run([fet_mul,fet_intermed])
    printTitle(tf.multiply)
    print(result)


#--------------------------------  Feed  --------------------------------
'''
上述示例在计算图中引入了 tensor, 以常量或变量的形式存储.
TensorFlow 还提供了 feed 机制,
该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.

feed 使用一个 tensor 值临时替换一个操作的输出结果.
你可以提供 feed 数据作为 run() 调用的参数.
feed 只在调用它的方法内有效, 方法结束, feed 就会消失.
最常见的用例是将某些特殊的操作指定为 "feed" 操作,
标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
'''
feed_intput1 = tf.placeholder(tf.float32)
feed_intput2 = tf.placeholder(tf.float32)
output = tf.multiply(feed_intput1, feed_intput2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={feed_intput1:111,feed_intput2:2}))


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#-----------------                                     ------------------
#-----------------                MNIST                ------------------
#-----------------                                     ------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
