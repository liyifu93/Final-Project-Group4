import numpy as np
from keras.models import load_model
import cv2
import os
from keras.utils import to_categorical
import random
import tensorflow as tf
from sklearn.metrics import classification_report

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

model_path = os.getcwd() + '/NonIdeal_Models/2/'
model = load_model(model_path + 'train_22PMU_MissingOneData_10dB_fullmodel_group4_2.hdf5')

img_path = os.getcwd() + '/Validation/Test_Model/CNN_40dB_800Mannual'


RESIZE_TO = 96
x, y = [], []

for file in os.listdir(img_path):
    print(file)
    for image in os.listdir(img_path + '/' + file + '/'):
        x.append(cv2.resize(cv2.imread(img_path + '/' + file + '/' + image, cv2.IMREAD_UNCHANGED),
                                  (RESIZE_TO, RESIZE_TO)))
        y.append(int(file))

x, y = np.array(x), np.array(y)
x = x / 255

# One hot encoding labels
y = to_categorical(y - 1)

test_pred = model.evaluate(x, y)
print('Test loss on test set:', test_pred[0])
print('Test accuracy on test set:', 100 * test_pred[1])

print(y.shape)

y_pred = np.argmax(model.predict(x), axis=1)
correct = np.where(y_pred == np.argmax(y, axis=1))[0]
print("Found %d correct labels" % len(correct))
