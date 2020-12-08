import os
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Model, Input, activations
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# Set up

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Visualization function
def show_sns_image(X, Encoded, Recons,  n = 5, height = 28, width = 28, title =''):
    plt.figure(figsize=(10,5))
    for i in range(n):
        j = np.random.randint(0, len(X))
        print(j)
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X[j])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 6)
        plt.imshow(Encoded[j].reshape((height, width)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 11)
        plt.imshow(Recons[j])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #print("Error for image", i, "is", np.sum((X[j] - Recons[j]) ** 2))
    plt.suptitle(title, fontsize = 20)

# model path name
autoencoder_path = 'Ideal_Model/train_idealmodel_autoencoder_group4_3.hdf5'
model_path = 'Ideal_Model/train_idealmodel_fullmodel_group4_3.hdf5'


DATA_DIR = os.getcwd() + "/PMU_PU/"

RESIZE_TO = 96

x, y = [], []

for file in os.listdir(DATA_DIR):
    print(file)
    for image in os.listdir(DATA_DIR + '/' + file + '/'):
        x.append(cv2.resize(cv2.imread(DATA_DIR + '/' + file + '/' + image, cv2.IMREAD_UNCHANGED), (RESIZE_TO, RESIZE_TO)))
        y.append(int(file))

x, y = np.array(x), np.array(y)
print(x.shape, y.shape)

# Counting how many targets for each class
classes, counts = np.unique(y, return_counts=True, axis=0)
classes = classes.tolist()  # Converting to list
counts = counts.tolist()  # Converting to list
print(classes, counts)

# Normalizing the images
x = x / 255

# Splitting the data

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size= 0.20, random_state= 0)


xtrain, xval, train_ground, validate_ground = train_test_split(x_train_val, x_train_val, test_size= 0.125, random_state= 0)
print(x_train_val.shape, x_test.shape)
print(xtrain.shape, xval.shape)
print(train_ground.shape, validate_ground.shape)


#xtrain = np.reshape(xtrain, (len(xtrain), RESIZE_TO, RESIZE_TO, 4))
#xtest = np.reshape(xtest, (len(xtest), RESIZE_TO, RESIZE_TO, 4))
#ytrain, ytest = to_categorical(ytrain, num_classes=8), to_categorical(ytest, num_classes= 8)


# Hyper parameters
N_EPOCHS = 30
Batch_size = 512
LR = 0.01
DROPOUT = 0.5

########################################################################################################################
# building the model
input_image = Input(shape=(96, 96, 4))
### Downsampling ---- Encoder
print('-- Encoding --')
z = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input_image) # shape 96 x 96
z = layers.BatchNormalization()(z)
z = layers.MaxPool2D((2,2))(z) # shape 48 x 48

z = layers.Conv2D(32, (3,3), padding='same')(z) # shape 48 x 48
z = layers.BatchNormalization()(z)
z = layers.MaxPool2D((2,2))(z) # shape 24 x 24
z = activations.relu(z)

z = layers.Conv2D(64, (3,3), padding='same', activation='relu')(z) # 24 x 24
z = layers.BatchNormalization()(z)
encoder = layers.MaxPool2D((2,2))(z) # shape 12 x 12

### Upsampling ---- Decoder
print('-- Decoding')
z = layers.Conv2D(64, (3, 3), padding ='same', activation='relu')(encoder) # shape 12 x 12
z = layers.BatchNormalization()(z)
z = layers.UpSampling2D((2,2))(z) # shape 24 x 24

z = layers.Conv2D(32, (3, 3), padding ='same')(z) # shape 24 x 24
z = layers.BatchNormalization()(z)
z = layers.UpSampling2D((2,2))(z) # shape 48 x 48
z = activations.relu(z)

z = layers.Conv2D(16, (3, 3), padding ='same', activation='relu')(z) # shape 96 x 96
z = layers.BatchNormalization()(z)
z = layers.UpSampling2D((2,2))(z) # shape 48 x 48

# 4 channels because we have 4 channels in the input
decoder = layers.Conv2D(4, (3, 3), activation = 'sigmoid', padding = 'same')(z) # shape 48 x 48

# Building the model
autoencoder = Model(input_image, decoder)

# Printing the model summary
print(autoencoder.summary())
########################################################################################################################


# Callbacks for fitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
model_check_point = ModelCheckpoint(filepath=autoencoder_path, save_best_only=True, monitor="val_loss")

# Compiling the autoencoder model
autoencoder.compile(optimizer=Adam(learning_rate= LR), loss='mse')

# Fitting the autoencoder model
# WATCH OUT THE OUTPUT
history_autoencoder = autoencoder.fit(xtrain, train_ground, epochs= N_EPOCHS, batch_size= Batch_size, validation_data=(xval, validate_ground),
                callbacks=[early_stop, model_check_point])

# Saving the auto encoder model
autoencoder.save(autoencoder_path)

# Making model to get the encoded representation
get_encoder = Model(autoencoder.input, autoencoder.get_layer('max_pooling2d_2').output)

# Getting the encoded sns heatmaps
encoded_heatmap = get_encoder.predict(xval)
encoded_heatmap = encoded_heatmap.reshape((len(xval), 12*12*64)) # depends on the channel
print(encoded_heatmap.shape)

# Getting the reconstructed sns heatmaps
reconstructed_maps = autoencoder.predict(xval)

# Visualizing data
show_sns_image(xval, encoded_heatmap, reconstructed_maps, height= 96, width= 96, title='Test Images - Encoded Test Images - Reconstructed Test Images')
plt.savefig('Ideal_Model/Autoencoder_Test_idealmodel')
plt.show()

# Plotting training error and validation error for the autoencoder model
plt.loglog(history_autoencoder.history['loss'], label='training')
plt.loglog(history_autoencoder.history['val_loss'], label='validation')
plt.title('Training loss vs Validation loss for Autoencoder')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(b = True, which = 'both')
plt.legend()
plt.savefig('Ideal_Model/Autoencoder_Loss_idealmodel')
plt.show()


# splitting data to build the classifier model
x_train, x_val, y_train, y_val = train_test_split(x_train_val,y_train_val,test_size=0.2,random_state=0)
# One hot encoding labels
y_test = to_categorical(y_test - 1)
y_train = to_categorical(y_train - 1)
y_val = to_categorical(y_val - 1)
print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# Classification Model
# --- First we get the same encoding layers from the autoencoder ---
z = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input_image) # shape 96 x 96
z = layers.BatchNormalization()(z)
z = layers.MaxPool2D((2,2))(z) # shape 48 x 48

z = layers.Conv2D(32, (3,3), padding='same')(z) # shape 48 x 48
z = layers.BatchNormalization()(z)
z = layers.MaxPool2D((2,2))(z) # shape 24 x 24
z = activations.relu(z)

z = layers.Conv2D(64, (3,3), padding='same', activation='relu')(z) # 24 x 24
z = layers.BatchNormalization()(z)
encoder = layers.MaxPool2D((2,2))(z) # shape 12 x 12

# Then, we flatten the encoder layer and add some fully connected layers to classify the 8 classes (0 - 7)
flat = layers.Flatten()(encoder)
den = layers.Dense(128, activation='relu')(flat)
drop = layers.Dropout(DROPOUT)(den)
#den = layers.Dense(128, activation='relu')(den)
out = layers.Dense(8, activation='softmax')(drop)


# Hyper parameters
N_EPOCHS = 10 # The remaining hyper-parameters are the same (learning rate and batch size)

# Building the model
full_model = Model(input_image, out)

# Printing the model summary
print(full_model.summary())

# Assigned weights to the encoding layers
for l1,l2 in zip(full_model.layers[:11],autoencoder.layers[0:11]):
    l1.set_weights(l2.get_weights())

# Freezing the encoding layers
for layer in full_model.layers[0:11]:
    layer.trainable = False

# Compiling the classification model
full_model.compile(optimizer=Adam(learning_rate= LR) , loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the classification model
history_classification = full_model.fit(x_train, y_train, batch_size= Batch_size,epochs= N_EPOCHS,validation_data=(x_val, y_val))

# Plotting training error and validation error
plt.loglog(history_classification.history['loss'], label='training')
plt.loglog(history_classification.history['val_loss'], label='validation')
plt.title('Training Loss vs Validation Loss for Full Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(b = True, which = 'both')
plt.legend()
plt.savefig('Ideal_Model/FullModel_Loss_idealmodel')
plt.show()

plt.plot(history_classification.history['accuracy'], label='training')
plt.plot(history_classification.history['val_accuracy'], label='validation')
plt.title('Training Accuracy vs Validation Accuracy for Full Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(b = True, which = 'both')
plt.legend()
plt.savefig('Ideal_Model/FullModel_Accuracy_idealmodel')
plt.show()


# Testing the last model
test_pred = full_model.evaluate(x_test, y_test)
print('Test loss on test set:', test_pred[0])
print('Test accuracy on test set:', test_pred[1])

# Predicting labels
class_pred = full_model.predict(x_test)
class_pred = np.argmax(np.round(class_pred),axis=1)
print(class_pred.shape, y_test.shape)

# Counting the correctly labeled images and plot some images that were correctly classified
correct = np.where(class_pred == np.argmax(y_test, axis=1))[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct])
    plt.title("Predicted {}, Class {}".format(class_pred[correct], np.argmax(y_test, axis=1)[correct]))
plt.savefig('Ideal_Model/FullModel_Correctly_Labeled_idealmodel')
plt.show()

# Counting the incorrectly labeled images and plot some images that were incorrectly classified
incorrect = np.where(class_pred != np.argmax(y_test, axis=1))[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect])
    plt.title("Predicted {}, Class {}".format(class_pred[incorrect], np.argmax(y_test, axis=1)[incorrect]))
plt.savefig('Ideal_Model/FullModel_Incorrectly_Labeled_idealmodel')
plt.show()


# Printing the classification report of the full model
class_names = ["Class {}".format(i) for i in range(8)]
print(classification_report(np.argmax(y_test, axis=1), class_pred, target_names=class_names))
# Saving the full model
full_model.save(model_path)


'''# Hyper parameters
N_EPOCHS = 10 # The remaining hyper-parameters are the same (learning rate and batch size)

# Construct the full model
full_model_2 = Model(input_image, out)
# Unfreezing the frozen layers
#for layer in full_model_2.layers[0:11]:
 #   layer.trainable = True

# Compiling the full model (again but with all layers are UNFROZEN)
full_model_2.compile(optimizer=Adam(learning_rate= LR) , loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the classification model with all layers are unfrozen
history_classification_2 = full_model_2.fit(x_train, y_train, batch_size=Batch_size,epochs=N_EPOCHS,validation_data=(x_val, y_val))

# Saving the full model without freezing
full_model_2.save(model_path)

# Plotting training error vs validation error and training accuracy vs validation accuracy for the full model
plt.loglog(history_classification_2.history['loss'], label='training')
plt.loglog(history_classification_2.history['val_loss'], label='validation')
plt.title('Training Loss vs Validation Loss for Full Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(b = True, which = 'both')
plt.legend()
plt.savefig('Ideal_Model/FullModel_Loss_idealmodel')
plt.show()

plt.plot(history_classification_2.history['accuracy'], label='training')
plt.plot(history_classification_2.history['val_accuracy'], label='validation')
plt.title('Training Accuracy vs Validation Accuracy for Full Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(b = True, which = 'both')
plt.legend()
plt.savefig('Ideal_Model/FullModel_Accuracy_idealmodel')
plt.show()


# Testing the last model
test_pred = full_model_2.evaluate(x_test, y_test)
print('Test loss on test set:', test_pred[0])
print('Test accuracy on test set:', test_pred[1])

# Predicting labels
class_pred = full_model_2.predict(x_test)
class_pred = np.argmax(np.round(class_pred),axis=1)
print(class_pred.shape, y_test.shape)

# Counting the correctly labeled images and plot some images that were correctly classified
correct = np.where(class_pred == np.argmax(y_test, axis=1))[0]
print("Found %d correct labels" % len(correct))
plt.figure(figsize=(10,5))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct])
    plt.title("Predicted {}, Class {}".format(class_pred[correct], np.argmax(y_test, axis=1)[correct]))
plt.savefig('Ideal_Model/FullModel_Correctly_Labeled_idealmodel')
plt.show()

# Counting the incorrectly labeled images and plot some images that were incorrectly classified
incorrect = np.where(class_pred != np.argmax(y_test, axis=1))[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect])
    plt.title("Predicted {}, Class {}".format(class_pred[incorrect], np.argmax(y_test, axis=1)[incorrect]))
plt.savefig('Ideal_Model/FullModel_Incorrectly_Labeled_idealmodel')
plt.show()


# Printing the classification report of the full model
class_names = ["Class {}".format(i) for i in range(8)]
print(classification_report(np.argmax(y_test, axis=1), class_pred, target_names=class_names))'''