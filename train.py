import tensorflow as tf
import numpy as np
from utils import *
from model import res50
from tables import *
from keras.utils import to_categorical


#config = tf.ConfigProto()         #using GPU
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

EPOCH = 100
TRAIN_NUM =8000
VALID_NUM = 2000
BATCH_SIZE = 8
ITERATION_TRAIN = TRAIN_NUM // BATCH_SIZE
ITERATION_VALID = VALID_NUM // BATCH_SIZE
model = res50()
train_file = open_file("train_100_100.h5", mode="r", title="Test file")
#valid_file = open_file("val_100_100.h5", mode="r", title="Test files")
print("***********************************")
for epoch in range(EPOCH):
    for index in range(ITERATION_TRAIN):
        X , Y = read_train_data(train_file, index * BATCH_SIZE, (index+1) * BATCH_SIZE)
        X = X[:,:,:, np.newaxis]
        Y = to_categorical(Y, 100) 
        loss = model.train_on_batch(X, Y)
        print("epoch:" ,epoch, "index:", index, "loss:", loss)
