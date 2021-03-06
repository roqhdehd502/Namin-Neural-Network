import os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    Namin Neural Network.mk2
    -----------------
    conv1 - relu1 - pool1 - bn1
    conv2 - relu2 - pool2 - bn2
    conv3 - relu3 - pool3 -
    conv4 - relu4 - pool4 - bn4
    affine - softmax

    Parameters
    ----------
    Input Layer(Input Size): 100*100*3
    First Layer: Conv1(4EA, 4*4*3, Strides=2, Padding=VALID) - ReLU1
                - 49*49*4 - Pool1(2*2, Strides=1) - 48*48*4 - Bn1
    Second Layer: Conv2(16EA, 6*6*4, Strides=1, Padding=VALID) - ReLU2
                - 43*43*16 - Pool2(2*2, Strides=1) - 42*42*16 - Bn2
    Third Layer: Conv3(64EA, 9*9*16, Strides=3, Padding=VALID) - ReLU3
                - 12*12*64 - Pool3(2*2, Strides=1) - 11*11*64
    Fourth layer: Conv4(16EA, 10*10*64, Strides=1, Padding=VALID) - ReLU4
                - 2*2*16 - Pool4(2*2, Strides=1) - 1*1*16 - Bn3
    Output Layer: Affine(W=1*1*16, B=16) - Output Nodes = 3(Frank, Mike, T)
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
        # 1st Layer(Conv1 - relu1 - maxpool1 - bn1) = 48*48*4
        conv1 = tf.layers.conv2d(X, 4, 4, (2, 2), padding='VALID', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 2, (1, 1), padding='VALID')
        bn1 = tf.compat.v1.layers.batch_normalization(pool1, training=True)
        # 2nd Layer(Conv2 - relu2 - maxpool2 - bn2) = 42*42*16
        conv2 = tf.layers.conv2d(bn1, 16, 6, (1, 1), padding='VALID', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 2, (1, 1), padding='VALID')
        bn2 = tf.compat.v1.layers.batch_normalization(pool2, training=True)
        # 3rd Layer(Conv3 - relu3 - maxpool3) = 11*11*64
        conv3 = tf.layers.conv2d(bn2, 64, 9, (3, 3), padding='VALID', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, 2, (1, 1), padding='VALID')
        # 4th Layer(Conv4 - relu4 - maxpool4 - bn3) = 1*1*16
        conv4 = tf.layers.conv2d(pool3, 16, 10, (1, 1), padding='VALID', activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4, 2, (1, 1), padding='VALID')
        bn3 = tf.compat.v1.layers.batch_normalization(pool4, training=True)
        # Fully Connected Layer(Affine)
        affine1 = tf.layers.flatten(bn3)
        # Output Layer
        output = tf.layers.dense(affine1, num_class)
        
    # Softmax with Loss
    with tf.variable_scope('Loss'):
        Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y, logits=output))
    
    # Training with Adam    
    train_step = tf.train.AdamOptimizer(0.001).minimize(Loss) 
    saver = tf.train.Saver()

# Size
np.sum([np.product(var.shape) for var in g.get_collection('trainable_variables')]).value

# Setting Batch with Training
batch_size = 1461
epoch = 1000

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch_data, batch_label = batch(trainlist, batch_size)     
        _, loss = sess.run([train_step, Loss], feed_dict = {X: batch_data, Y: batch_label})
        print("Epoch:",i,"Loss:",loss)

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
