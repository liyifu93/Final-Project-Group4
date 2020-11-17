import os
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
# matplotlib inline
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# https://www.datacamp.com/community/tutorials/autoencoder-classifier-python#
# https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial#cae
# %% ----------------------------------- environment -------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # model will be trained on GPU 0

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

def encoder(input_img):
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)    # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)    # 14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)    # 7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)    # 7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)    # 7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

# %% ---------------------------- Loading & Exploration ----------------------------------------------------------------
train_data = extract_data('/home/ubuntu/ML2_Final/MNIST/train-images-idx3-ubyte.gz', 60000)
test_data = extract_data('/home/ubuntu/ML2_Final/MNIST/t10k-images-idx3-ubyte.gz', 10000)

train_labels = extract_labels('/home/ubuntu/ML2_Final/MNIST/train-labels-idx1-ubyte.gz', 60000)
test_labels = extract_labels('/home/ubuntu/ML2_Final/MNIST/t10k-labels-idx1-ubyte.gz', 10000)

# Shapes of training set
# print("Training set (images) shape: {shape}".format(shape=train_data.shape))    # (60000, 28, 28)

# Shapes of test set
# print("Test set (images) shape: {shape}".format(shape=test_data.shape))    # (10000, 28, 28)

# Create dictionary of target classes
label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
# print('Original label:', train_labels[0])
# print('After conversion to one-hot:', train_Y_one_hot[0])

plt.figure(figsize=[5, 5])
# Display the first image in training data
# plt.subplot(121)
# curr_img = np.reshape(train_data[10], (28, 28))
# curr_lbl = train_labels[10]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
# plt.subplot(122)
# curr_img = np.reshape(test_data[10], (28, 28))
# curr_lbl = test_labels[10]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# %% ---------------------------------- Data Processing ----------------------------------------------------------------
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
# print(train_data.shape, test_data.shape)    # ((60000, 28, 28, 1), (10000, 28, 28, 1))
# print(train_data.dtype, test_data.dtype)    # (dtype('float32'), dtype('float32'))
# print(np.max(train_data), np.max(test_data))    # (255.0, 255.0)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)
# print(np.max(train_data), np.max(test_data))    # (1.0, 1.0)

# train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
#                                                              train_data,
#                                                              test_size=0.2,
#                                                              random_state=13)

train_X, valid_X, train_label, valid_label = train_test_split(train_data,
                                                              train_Y_one_hot,
                                                              test_size=0.2,
                                                              random_state=13
                                                              )
# print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)    # ((48000, 28, 28, 1), (12000, 28, 28, 1), (48000, 10), (12000, 10))

# %% ----------------------------- Convolutional Autoencoder -----------------------------------------------------------
batch_size = 64
epochs = 100   # 200
inChannel = 1
x, y = 28, 28
input_img = Input(shape=(x, y, inChannel))
num_classes = 10

# autoencoder = Model(input_img, decoder(encoder(input_img)))

# autoencoder.compile(loss='mean_squared_error',
#                     optimizer=RMSprop()
#                     )

# autoencoder.summary()

encode = encoder(input_img)
full_model = Model(input_img, fc(encode))

# for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[0:19]):
#     l1.set_weights(l2.get_weights())

# print(autoencoder.get_weights()[0][1])
# print(full_model.get_weights()[0][1])

for layer in full_model.layers[0:19]:
    layer.trainable = True     # False - Classification

full_model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy']
                   )

full_model.summary()

# %% ----------------------------- Training ----------------------------------------------------------------------------
# autoencoder_train = autoencoder.fit(train_X,
#                                     train_ground,
#                                     batch_size=batch_size,
#                                     epochs=epochs,
#                                     verbose=1,
#                                     validation_data=(valid_X, valid_ground)
#                                     )

classify_train = full_model.fit(train_X,
                                train_label,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(valid_X, valid_label)
                                )

# loss = autoencoder_train.history['loss']
# val_loss = autoencoder_train.history['val_loss']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']

epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# %% ----------------------------- Evaluation --------------------------------------------------------------------------
test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
# print('Test loss:', test_eval[0])       # ('Test loss:', 0.7068972043234281)
# print('Test accuracy:', test_eval[1])   # ('Test accuracy:', 0.9205)

predicted_classes = full_model.predict(test_data)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
# print(predicted_classes.shape, test_labels.shape)    # ((10000,), (10000,))

correct = np.where(predicted_classes == test_labels)[0]
print('Found correct labels: ', len(correct))    # 9204
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_data[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != test_labels)[0]
print('Found incorrect labels: ', len(incorrect))    # 796
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_data[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

# %% ----------------------------- Save Model --------------------------------------------------------------------------
# autoencoder.save_weights('autoencoder.h5')
# full_model.save_weights('autoencoder_classification.h5')
# full_model.save_weights('classification_complete.h5')