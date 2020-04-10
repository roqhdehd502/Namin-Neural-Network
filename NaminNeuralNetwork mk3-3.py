import os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    Namin Neural Network.mk3
    -----------------
    conv1 - relu1 - pool1 - bn1
    conv2 - relu2 - pool2 - bn2
    conv3 - relu3 - pool3 - bn3
    affine - softmax

    Parameters
    ----------
    Input Layer(Input Size): 100*100*3
    First Layer: Conv1(3EA, 3*3*3, Strides=3, Padding=1) - ReLU1
                - 34*34*3 - Pool1(2*2, Strides=1) - 33*33*3 - Bn1
    Second Layer: Conv2(6EA, 6*6*3, Strides=1, Padding=VALID) - ReLU2
                - 28*28*6 - Pool2(2*2, Strides=2) - 14*14*6 - Bn2
    Third Layer: Conv3(9EA, 3*3*6, Strides=3, Padding=2) - ReLU3
                - 6*6*9 - Pool3(2*2, Strides=2) - 3*3*9 - Bn3
    Output Layer: Affine(W=3*3*9, B=9) - Output Nodes = 3(Frank, Mike, T)
 """
 
 # Load to Franklin, Michael, Trevor Images
trainlist, testlist = [], []
with open('train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        trainlist.append([tmp[0], tmp[1]])
        
with open('test.txt') as f:
    for line in f:
        tmp = line.strip().split()
        testlist.append([tmp[0], tmp[1]])
        
# Image Preprocessing
IMG_H = 100
IMG_W = 100
IMG_C = 3

def readimg(path):
    img = plt.imread(path)
    return img

def batch(path, batch_size):
    img, label, paths = [], [], []
    for i in range(batch_size):
        img.append(readimg(path[0][0]))
        label.append(int(path[0][1]))
        path.append(path.pop(0))
        
    return img, label
    
# Neural Network
num_class = 3 # Franklin Clinton, Trevor Philips, Michael De Santa

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C]) # Input Layer
    Y = tf.placeholder(tf.int32, [None])
    
    with tf.variable_scope('CNN'):
        # 1st Layer(Conv1 - relu1 - maxpool1 - bn1) = 33*33*3
        conv1 = tf.layers.conv2d(X, 3, 3, (3, 3), padding='SAME', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 2, (1, 1), padding='VALID')
        bn1 = tf.compat.v1.layers.batch_normalization(pool1, training=True)
        # 2nd Layer(Conv2 - relu2 - maxpool2 - bn2) = 14*14*6
        conv2 = tf.layers.conv2d(bn1, 6, 6, (1, 1), padding='VALID', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 2, (2, 2), padding='VALID')
        bn2 = tf.compat.v1.layers.batch_normalization(pool2, training=True)
        # 3rd Layer(Conv3 - relu3 - maxpool3 - bn3) = 3*3*9
        conv3 = tf.layers.conv2d(bn2, 9, 3, (3, 3), padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, 2, (2, 2), padding='VALID')
        bn3 = tf.compat.v1.layers.batch_normalization(pool3, training=True)
        # Fully Connected Layer(Affine)
        affine1 = tf.layers.flatten(bn3)
        # Output Layer
        output = tf.layers.dense(affine1, num_class)
        
    # Softmax with Loss
    with tf.variable_scope('Loss'):
        Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y, logits=output))
    
    # Training with Adam    
    train_step = tf.train.AdamOptimizer(0.005).minimize(Loss) 
    saver = tf.train.Saver()
    
    tf.summary.scalar('Epoch-Loss', Loss)
    merged = tf.summary.merge_all()
    
# Size
np.sum([np.product(var.shape) for var in g.get_collection('trainable_variables')]).value

# Setting Batch with Training
batch_size = 1261
epoch = 1000

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('tf_board', sess.graph)
    for i in range(epoch):
        batch_data, batch_label = batch(trainlist, batch_size)     
        _, loss, summary = sess.run([train_step, Loss, merged], feed_dict = {X: batch_data, Y: batch_label})
        print("Epoch:",i+1,"Loss:",loss)
        if i % 10 == 0:
            summary_writer.add_summary(summary, i)
            saver.save(sess, 'logs/model.ckpt', global_step = i+1)
        elif i+1 == epoch:
            summary_writer.add_summary(summary, i)
            saver.save(sess, 'logs/model.ckpt', global_step = i+1)
            
# Print an Accuracy
acc = 0

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint('logs')
    if checkpoint:
        saver.restore(sess, checkpoint)
    for i in range(len(testlist)):
        batch_data, batch_label = batch(testlist, 1)
        logit = sess.run(output, feed_dict = {X:batch_data})
        if np.argmax(logit[0]) == batch_label[0]:
            acc += 1
        else:
            print(logit[0], batch_label[0])
            
    print("Accuracy:", acc/len(testlist))
