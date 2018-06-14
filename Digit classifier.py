import numpy as np
import os

import tensorflow as tf
from tensorflow.python.framework import ops
import cv2

FILE_NAME = os.path.abspath('..\\course one\\PA2\\train.csv')
TMP_FILE_NAME = os.path.abspath("..\\course two\\PA Tensorflow\\temp_data.csv")


def load_images(file_name):
    data = np.genfromtxt(file_name,delimiter = ',')
    
    print(data.shape)
    data = data[1:]
    y = data[:,:1]
    x = data[:,1:]
    print(x.shape,y.shape)

    return x,y



def get_data(file_name):
        
    print("loading images")
    x, y = load_images(file_name)
    print("Shape while loading before file: ", y.shape)
    y = y.reshape((-1))
    
    labels = tf.one_hot(y, 10, axis = -1)
    sess = tf.Session()
    res = sess.run(labels)
    print("Shape while loading file y : ", res.shape)
    
    return x,res

def create_placeholders(n_h, n_w, n_c, n_y):
    '''
    n_h = height os the input image
    n_w = width of the image
    n_y = no of classes
    n_c = no of channels
    Returns
    x = placeholder for the input data
    y = placeholder for the input labels
    '''
    x = tf.placeholder(tf.float32, [None, n_h, n_w, n_c])
    y = tf.placeholder(tf.float32, [None, n_y])

    return x, y


def init_parameters():

    '''
    return dictionary of tensor containing w1 and w2
    '''
    tf.set_random_seed(1)
    w1 = tf.get_variable("w1", [4, 4, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    w2 = tf.get_variable("w2", [4, 4, 8, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    w3 = tf.get_variable("w3", [3, 3, 12, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    w4 = tf.get_variable("w4", [4, 4, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    w5 = tf.get_variable("w5", [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))


    parameters = {
        "w1" : w1,
        "w3" : w3,
        "w4" : w4,
        "w5" : w5,
        "w2" : w2}
    return parameters




def forward_prop(x, parameters):
    '''
    implements the forward prop for the model
    conv2d -> relu -> maxpool -> conv2d -> relu -> max_pool -> flatten -> fully_connected layer
    -> classes

    input :
     x : input data
     paramters :python dictionary containnig w1, w2 the shapes are given in init paramters
    returns:
    z3 = the output of the last linear unit
    '''
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    w3 = parameters["w3"]
    w4 = parameters["w4"]
    w5 = parameters["w5"]
    z1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'SAME')

    a1 = tf.nn.relu(z1)

    p1 = tf.nn.max_pool(a1, ksize = [1, 4, 4, 1],strides = [1, 4, 4, 1],  padding = "SAME")



    z2 = tf.nn.conv2d(p1, w2, strides = [1, 1, 1, 1], padding = "SAME")

    a2 = tf.nn.relu(z2)

    p2 = tf.nn.max_pool(a2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')



    z3 = tf.nn.conv2d(p2, w3, strides = [1, 1, 1, 1], padding = "SAME")

    a3 = tf.nn.relu(z3)

    p3 = tf.nn.max_pool(a3, ksize = [1, 3, 3, 1], strides = [1, 3, 3, 1], padding = 'SAME')



    z4 = tf.nn.conv2d(p3, w4, strides = [1, 1, 1, 1], padding = "SAME")

    a4 = tf.nn.relu(z4)

    p4 = tf.nn.max_pool(a4, ksize = [1, 4, 4, 1], strides = [1, 3, 3, 1], padding = 'SAME')


    z5 = tf.nn.conv2d(p4, w5, strides = [1, 1, 1, 1], padding = "SAME")

    a5 = tf.nn.relu(z5)

    p5 = tf.nn.max_pool(a5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    
    p = tf.contrib.layers.flatten(p5)

    z3 = tf.contrib.layers.fully_connected(p, 10, activation_fn = tf.sigmoid)

    
    return z3

def compute_cost(z3, y):
    
    #return the cost of predicted output
    
    print("v")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits  = z3, labels = y))
    return cost


def model( x_train, y_train, x_test, y_test,learning_rate = 0.01, epochs = 1000, print_cost = True):

    '''
    x_train.shape, x_test =  (?, 64, 64, 3)
    y_train.shape, y_train = (?, number_classes = 6)
    
    '''
    tf.set_random_seed(1)
    (m, n_h, n_w, n_c) = x_train.shape
    n_y = y_train.shape[1]
    x, y = create_placeholders(n_h, n_w, n_c, n_y)
    parameters = init_parameters()
    z3 = forward_prop(x, parameters)

    cost = compute_cost(z3, y_train)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    predict_output = tf.argmax(z3, 1)
    correct_prediction = tf.equal(predict_output,tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()
    saver  =  tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(epochs):
            
            _, tmp_cost = sess.run([optimizer, cost], feed_dict = {x : x_train, y: y_train})
            if  print_cost == True:
                print("epoch_no: ", epoch, "cost: ", tmp_cost)
            if epoch % 10 == 0:
                save_path = saver.save(sess, "check_points/model.ckpt")
                print("model saved at ", save_path)
        
    with tf.Session() as sess:
        saver.restore(sess, "check_points/model.ckpt" )
        print("model restored !!")
        predict = sess.run([z3], feed_dict = {x : x_test, y: y_test})
        print(predict)
        
    


    

def predict(image):

    img = cv2.imread(image)
    
    cv2.imshow("ajay", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("img.shape: ", img.shape)
    ret, img = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)
    print(img)
    img = cv2.resize(img, (28, 28))
    img = (255 - img)/255
    cv2.imshow("img", img)
    print(img[14])
    img = np.reshape(img, (-1, 28, 28,1))
    print(img.shape)
    x, y = create_placeholders(28, 28, 1, 10)
    parameters = init_parameters()
    z3 = forward_prop(x, parameters)
    saver  =  tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "check_points/model.ckpt" )
        z3 = sess.run([z3], feed_dict = {x : img})
        
        print(tf.argmax(z3[0][0]).eval())


def split_data(x, y, ratio):
    length = int(len(x)* ratio)
    x_train = x[:length]
    y_train = y[:length]
    x_test = x[length:]
    y_test = y[length:]
    return x_train, y_train, x_test, y_test

    



print("satrting the session of actual model !!")
x, y = get_data(FILE_NAME)
x = np.reshape(x , (-1, 28, 28, 1))
tmp = x
tmp = tmp.reshape(-1,28,28)
print(y[:10])
for i in range(10):
    cv2.imshow("image"+str(i), tmp[i])
x_train, y_train, x_test, y_test = split_data(x, y, 0.8)
print(y_train.dtype)
model(x_train, y_train, x_test, y_test)

print("ajay")
predict("nine.jpg")

