#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# # Building MLPs, CNNs,  and Generative Models with TensorFlow

# ## Task 1: Learn the basics of Keras API for TensorFlow

# In[1]:


from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt


# ### Experiment with official repository example mnist_mlp.py

# In[3]:


batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# In[4]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ### Experiment with official repository example mnist_cnn.py

# In[5]:


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# In[7]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ### Fashion MNIST MLP

# In[2]:


batch_size = 128
num_classes = 10
epochs = 30

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# #### MLP: Baseline (Example in the textbook)

# In[4]:


K.clear_session()
model = Sequential()
model.add(Dense(300, activation="relu", input_shape=(784,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()


# In[5]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# #### Experiments with MLP network

# In[3]:


def mlp_models(num_layers=3, arg_dict=arg_dict, dropout=False, dropout_lst=[], optimizer=RMSprop(), batch=128, info=""):
    K.clear_session()
    print(f"{info}")
    model = keras.models.Sequential()
    model.add(Input(shape=(784,)))
    for i in range(num_layers):
        model.add(Dense(arg_dict[i][0], 
                     activation=arg_dict[i][1], 
                     kernel_initializer=arg_dict[i][2], 
                     kernel_regularizer=arg_dict[i][3]))
        if dropout and i != (num_layers - 1):
            model.add(Dropout(dropout_lst[i]))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])   
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./temp.hdf5",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)
    
    history = model.fit(x_train, y_train,
                    batch_size=batch,
                    epochs=50,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback, early_callback])
    model.load_weights("./temp.hdf5")
    score = model.evaluate(x_test, y_test, verbose=0) 
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history.history, score


# ##### Architecture: Adding / removing layers

# In[6]:


arg_dict = [{0: (512, "relu", "glorot_uniform", None),
             1: (10, "softmax", "glorot_uniform", None),
            }, 
            {0: (512, "relu", "glorot_uniform", None),
             1: (128, "relu", "glorot_uniform", None),
             2: (10, "softmax", "glorot_uniform", None),
            },
            {0: (512, "relu", "glorot_uniform", None),
             1: (128, "relu", "glorot_uniform", None),
             2: (64, "relu", "glorot_uniform", None),
             3: (10, "softmax", "glorot_uniform", None),
            }]
history_lst = list()
info_lst = ["Two layers", "Three layers", "Four layers"]
for trial in range(3):
    his, _ = mlp_models(num_layers=trial + 2, arg_dict=arg_dict[trial], info=info_lst[trial])
    history_lst.append(his)


# In[14]:


plt.rcParams["figure.figsize"] = (8, 5)
for index in range(len(history_lst)):
    acc = history_lst[index]["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{index + 2} layers")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# ##### Architecture: Number of units

# In[19]:


arg_dict = [{0: (512, "relu", "glorot_uniform", None),
             1: (10, "softmax", "glorot_uniform", None),
            }, 
           {0: (256, "relu", "glorot_uniform", None),
            1: (10, "softmax", "glorot_uniform", None),
            }, 
           {0: (128, "relu", "glorot_uniform", None),
            1: (10, "softmax", "glorot_uniform", None),
            },]
history_lst = list()
info_lst = ["512", "256", "128"]
for trial in range(3):
    his, _ = mlp_models(num_layers=2, arg_dict=arg_dict[trial], info=info_lst[trial])
    history_lst.append(his)


# In[20]:


for index in range(len(history_lst)):
    acc = history_lst[index]["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{info_lst[index]}")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# ##### Initializations / Activations / Optimizers / Regularizations

# In[11]:


# No dropout
initialization_lst = [tf.keras.initializers.GlorotUniform(),
                      tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                      tf.keras.initializers.GlorotNormal(),
                     ]
initialization_info = ["glorot uniform", "random normal", "glorot normal"]
activation_lst = ["relu", "sigmoid", "tanh"]
optimizer_lst = [tf.keras.optimizers.RMSprop(),
                 tf.keras.optimizers.Adam(),
                 tf.keras.optimizers.Adam(learning_rate=0.01),
                 ]
optimizer_info = ["RMSprop", "adam default", "adam 0.01"]
regularization_lst = [regularizers.L1(l1=0.01), regularizers.L2(l2=0.01), None]
regularization_info = ["L1", "L2", "None"]

arg_dict = list()
for a in range(3):
    initialization = initialization_lst[a]
    for b in range(3):
        activation = activation_lst[b]
        for c in range(3):
            optimizer = optimizer_lst[c]
            for d in range(3):
                regularization = regularization_lst[d]
                info = " / ".join([initialization_info[a], activation_lst[b], optimizer_info[c], regularization_info[d]])
                arg_dict.append({0: (128, activation, initialization, regularization),
                                 1: (10, "softmax", initialization, None),
                                 "op": optimizer,
                                 "info": info},)
print(f"Parameter Settings: {len(arg_dict)}.")
print(f"Example: {arg_dict[0]}")


# In[12]:


score_lst = list()
records = list()
for trial in range(len(arg_dict)):
    _, score = mlp_models(num_layers=2, arg_dict=arg_dict[trial], optimizer=arg_dict[trial]["op"], info=arg_dict[trial]["info"])
    records.append(arg_dict[trial]["info"])
    score_lst.append(score)


# In[19]:


best = np.argmax(np.array(score_lst), axis=0)[1]
print(f"Best Settings: {records[best]}.")
print(f"Best acc: {score_lst[best][1]}.")


# In[22]:


# With dropout
## Setting 1
dropout_r = [0.1, 0.3, 0.5]
arg_dict = {0: (128, "sigmoid", "glorot_uniform", None),
            1: (10, "softmax", "glorot_uniform", None),
            }
info_lst = ["0.1", "0.3", "0.5"]
for trial in range(3):
    mlp_models(num_layers=2, arg_dict=arg_dict, dropout=True, dropout_lst=dropout_r, info=info_lst[trial])


# In[24]:


## Setting 2
dropout_r = [0.1, 0.3, 0.5]
arg_dict = {0: (512, "relu", "glorot_uniform", None),
            1: (10, "softmax", "glorot_uniform", None),
            }
info_lst = ["0.1", "0.3", "0.5"]
for trial in range(3):
    mlp_models(num_layers=2, arg_dict=arg_dict, dropout=True, dropout_lst=dropout_r, info=info_lst[trial])


# In[25]:


## Setting 3
dropout_r = [0.1, 0.3, 0.5]
arg_dict = {0: (128, "relu", "glorot_uniform", None),
            1: (10, "softmax", "glorot_uniform", None),
            }
info_lst = ["0.1", "0.3", "0.5"]
for trial in range(3):
    mlp_models(num_layers=2, arg_dict=arg_dict, dropout=True, dropout_lst=dropout_r, info=info_lst[trial])


# ### Fashion MNIST CNN

# #### CNN: Baseline (Example in the textbook)

# In[2]:


batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[7]:


model = keras.models.Sequential([ 
  keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
  keras.layers.MaxPooling2D(2),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"), 
  keras.layers.Dropout(0.5), 
  keras.layers.Dense(64, activation="relu"), 
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation="softmax") 
])

model.summary()


# In[9]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# #### Experiments with CNN network

# ##### Architecture: Adding / removing layers

# In[6]:


batch_size = 128
epochs = 50

model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath="./temp.hdf5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_callback1 = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

model1 = keras.models.Sequential([ 
  keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
  keras.layers.MaxPooling2D(2),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"), 
  keras.layers.Dropout(0.5), 
  keras.layers.Dense(64, activation="relu"), 
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation="softmax") 
])

model1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history1 = model1.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback1, early_callback1])
model1.load_weights("./temp.hdf5")
score = model1.evaluate(x_test, y_test, verbose=0)
print('Baseline')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Remove layers
model2 = keras.models.Sequential([ 
  keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
  keras.layers.MaxPooling2D(2),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"), 
  keras.layers.Dropout(0.5), 
  keras.layers.Dense(64, activation="relu"), 
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation="softmax") 
])

model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath="./temp.hdf5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_callback2 = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

model2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history2 = model2.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback2, early_callback2])
model2.load_weights("./temp.hdf5")
score = model2.evaluate(x_test, y_test, verbose=0)
print('Remove layers')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Add layers
model3 = keras.models.Sequential([ 
  keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
  keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
  keras.layers.MaxPooling2D(2),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"), 
  keras.layers.Dropout(0.5), 
  keras.layers.Dense(64, activation="relu"), 
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation="softmax") 
])

model_checkpoint_callback3 = tf.keras.callbacks.ModelCheckpoint(
    filepath="./temp.hdf5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_callback3 = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history3 = model3.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback3, early_callback3])
model3.load_weights("./temp.hdf5")
score = model3.evaluate(x_test, y_test, verbose=0)
print('Add layers')
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[9]:


plt.rcParams["figure.figsize"] = (8, 5)
history_lst = [history1.history, history2.history, history3.history]
info_lst = ["Baseline", "Remove layers", "Add layers"]

for index in range(len(history_lst)):
    acc = history_lst[index]["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{info_lst[index]}")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# ##### Architecture: Number of filters

# In[15]:


def cnn_models(filter_set=(64, 128, 256), dropout_set=(0.5, 0.5), optimizer=RMSprop(), info=""):
    print(info)
    K.clear_session()
    model = keras.models.Sequential([ 
      keras.layers.Conv2D(filter_set[0], 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
      keras.layers.MaxPooling2D(2),
      keras.layers.Conv2D(filter_set[1], 3, activation="relu", padding="same"),
      keras.layers.Conv2D(filter_set[1], 3, activation="relu", padding="same"), 
      keras.layers.MaxPooling2D(2), 
      keras.layers.Conv2D(filter_set[2], 3, activation="relu", padding="same"), 
      keras.layers.Conv2D(filter_set[2], 3, activation="relu", padding="same"), 
      keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"), 
      keras.layers.Dropout(dropout_set[0]), 
      keras.layers.Dense(64, activation="relu"), 
      keras.layers.Dropout(dropout_set[1]),
      keras.layers.Dense(10, activation="softmax") 
    ])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./temp.hdf5",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
    )

    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[model_checkpoint_callback, early_callback])
    model.load_weights("./temp.hdf5")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history, score


# In[16]:


filters = [(64, 128, 256), (32, 64, 128), (128, 256, 512)]
history_lst = list()
info_lst = ["(64, 128, 128, 256, 256)", "(32, 64, 64, 128, 128)", "(128, 256, 256, 512, 512)"]
for trial in range(3):
    his, _ = cnn_models(filter_set=filters[trial], info=info_lst[trial])
    history_lst.append(his)


# In[18]:


plt.rcParams["figure.figsize"] = (8, 5)
for index in range(len(history_lst)):
    acc = history_lst[index].history["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{info_lst[index]}")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# ##### Regularizations

# In[19]:


# Dropout rate
dropout_r = [(0.5, 0.5), (0.3, 0.3), (0.1, 0.1)]
history_lst = list()
info_lst = ["(0.5, 0.5)", "(0.3, 0.3)", "(0.1, 0.1)"]
for trial in range(3):
    his, _ = cnn_models(filter_set=(32, 64, 128), dropout_set=dropout_r[trial], info=info_lst[trial])
    history_lst.append(his)


# In[20]:


for index in range(len(history_lst)):
    acc = history_lst[index].history["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{info_lst[index]}")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# In[22]:


optimizer_lst = [tf.keras.optimizers.RMSprop(),
                 tf.keras.optimizers.Adam(),
                 tf.keras.optimizers.Adam(learning_rate=0.01),
                 ]
info_lst = ["RMSprop", "adam default", "adam 0.01"]
history_lst = list()
for trial in range(3):
    his, _ = cnn_models(filter_set=(32, 64, 128), dropout_set=(0.1, 0.1), optimizer=optimizer_lst[trial], info=info_lst[trial])
    history_lst.append(his)


# In[23]:


for index in range(len(history_lst)):
    acc = history_lst[index].history["val_accuracy"]
    plt.plot(list(range(1, len(acc) + 1)), acc, label=f"{info_lst[index]}")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("val_acc")
plt.title("Validation Acc")
plt.show()


# #### Top three hyperparameter settings

# **MLP**
# 
# 1. Test acc: 0.892. Two layers, units: (512, 10), activations: "relu", initializations: "glorot_uniform", optimizer: RMSprop, Dropout: 0.3.
# 
# 2. Test acc: 0.891. Two layers, units: (512, 10), activations: "relu", initializations: "glorot_uniform", optimizer: RMSprop, Dropout: None.
# 
# 3. Test acc: 0.889. Two layers, units: (128, 10), activations: "relu", initializations: "glorot_uniform", optimizer: RMSprop, Dropout: 0.1.
# 
# **CNN**
# 
# 1. Test acc: 0.918. filters: (32, 64, 64, 128, 128), optimizer: RMSprop, Dropout: (0.1, 0.1).
# 
# 2. Test acc: 0.912. filters: (32, 64, 64, 128, 128), optimizer: Adam, Dropout: (0.1, 0.1).
# 
# 3. Test acc: 0.911. filters: (32, 64, 64, 128, 128), optimizer: RMSprop, Dropout: (0.5, 0.5).

# ### CIFAR-10 MLP

# #### MLP: Baseline (Example in the textbook)

# In[2]:


batch_size = 128
num_classes = 10
epochs = 30

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[5]:


x_train = x_train.reshape(50000, -1)
x_test = x_test.reshape(10000, -1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[7]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[3072,]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()


# In[8]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# #### Experiments with MLP network

# In[12]:


def mlp_models(num_layers=3, arg_dict=arg_dict, dropout=False, dropout_lst=[], optimizer=RMSprop(), batch=128, info=""):
    K.clear_session()
    print(f"{info}")
    model = keras.models.Sequential()
    model.add(Input(shape=(3072,)))
    for i in range(num_layers):
        model.add(Dense(arg_dict[i][0], 
                     activation=arg_dict[i][1], 
                     kernel_initializer=arg_dict[i][2], 
                     kernel_regularizer=arg_dict[i][3]))
        if dropout and i != (num_layers - 1):
            model.add(Dropout(dropout_lst[i]))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])   
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./temp.hdf5",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)
    
    history = model.fit(x_train, y_train,
                    batch_size=batch,
                    epochs=50,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[model_checkpoint_callback, early_callback])
    model.load_weights("./temp.hdf5")
    score = model.evaluate(x_test, y_test, verbose=0) 
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history.history, score


# ##### Top three settings in Fashion MNIST

# In[13]:


arg_dict = [{0: (512, "relu", "glorot_uniform", None),
             1: (10, "softmax", "glorot_uniform", None),
            },
            {0: (512, "relu", "glorot_uniform", None),
             1: (10, "softmax", "glorot_uniform", None),
            },
            {0: (128, "relu", "glorot_uniform", None),
             1: (10, "softmax", "glorot_uniform", None),
            },
           ]
dropout_lst = [0.3, None, 0.1]
info_lst = ["Top 1", "Top 2", "Top 3"]

for trial in range(3):
    mlp_models(num_layers=2, arg_dict=arg_dict[trial], dropout=dropout_lst[trial], dropout_lst=[dropout_lst[trial]], 
               optimizer=RMSprop(), batch=128, info=info_lst[trial])


# ##### New model: Best settings

# In[16]:


arg_dict = {0: (512, "relu", "glorot_uniform", None),
            1: (128, "relu", "glorot_uniform", None),
            2: (32, "relu", "glorot_uniform", None),
            3: (10, "softmax", "glorot_uniform", None),
            }
_x, _y = mlp_models(num_layers=4, arg_dict=arg_dict, dropout=True, dropout_lst=[0.1, 0.1, 0.1], optimizer=RMSprop(), batch=128, info="Best model")


# ### CIFAR-10 CNN

# #### CNN: Baseline (Example in the textbook)

# In[3]:


batch_size = 128
num_classes = 10
epochs = 30

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)


# In[4]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[6]:


model = keras.models.Sequential([ 
  keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[32, 32, 3]),
  keras.layers.MaxPooling2D(2),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
  keras.layers.Conv2D(128, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.Conv2D(256, 3, activation="relu", padding="same"), 
  keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
  keras.layers.Dense(128, activation="relu"), 
  keras.layers.Dropout(0.5), 
  keras.layers.Dense(64, activation="relu"), 
  keras.layers.Dropout(0.5),
  keras.layers.Dense(10, activation="softmax") ])
model.summary()


# In[8]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# #### Experiments with CNN network

# In[9]:


def cnn_models(filter_set=(32, 64, 128), dropout_set=(0.1, 0.1), optimizer=RMSprop(), info=""):
    print(info)
    K.clear_session()
    model = keras.models.Sequential([ 
      keras.layers.Conv2D(filter_set[0], 7, activation="relu", padding="same", input_shape=[32, 32, 3]),
      keras.layers.MaxPooling2D(2),
      keras.layers.Conv2D(filter_set[1], 3, activation="relu", padding="same"),
      keras.layers.Conv2D(filter_set[1], 3, activation="relu", padding="same"), 
      keras.layers.MaxPooling2D(2), 
      keras.layers.Conv2D(filter_set[2], 3, activation="relu", padding="same"), 
      keras.layers.Conv2D(filter_set[2], 3, activation="relu", padding="same"), 
      keras.layers.MaxPooling2D(2), keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"), 
      keras.layers.Dropout(dropout_set[0]), 
      keras.layers.Dense(64, activation="relu"), 
      keras.layers.Dropout(dropout_set[1]),
      keras.layers.Dense(10, activation="softmax") 
    ])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./temp.hdf5",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
    )

    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=[model_checkpoint_callback, early_callback])
    model.load_weights("./temp.hdf5")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history, score


# ##### Top three settings in Fashion MNIST

# In[10]:


_x, _y = cnn_models(info="Top 1")
_x, _y = cnn_models(optimizer=tf.keras.optimizers.Adam(), info="Top 2")
_x, _y = cnn_models(dropout_set=(0.5, 0.5), info="Top 3")


# ##### New model: Best settings

# In[12]:


_x, _y = cnn_models(filter_set=(64, 128, 256), dropout_set=(0.1, 0.1), optimizer=tf.keras.optimizers.Adam(), info="Best model")


# ## Task 2: Develop a "Tell-the-time" network

# In[8]:


import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, Input, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras import regularizers
import keras
from keras import layers
import math


# In[2]:


image = np.load("images.npy")
label = np.load("labels.npy")
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")


# In[3]:


image = image / 255.0
image = image.reshape(image.shape + (1,))
plt.imshow(image[0])
plt.show()


# In[4]:


X, X_test, y, y_test = train_test_split(image, label, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape}")
print(f"Valid set: {X_valid.shape}")
print(f"Test set: {X_test.shape}")


# ### Regression

# In[6]:


# Task: Regression
def regression_model():
    in_put = Input((150, 150, 1))
    x = Conv2D(32, (3, 3), activation="relu")(in_put)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", strides=(2, 2))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1)(x)
    model = tf.keras.Model(in_put, x)
    model.summary()
    def common_sense_reg(y_true, y_pred):
        d = K.abs(y_true - y_pred)
        return (1 - (d // 6)) * (d) + (d // 6) * (12 - d)
    model.compile(optimizer="adam", loss="mse", metrics=common_sense_reg)
    return model

def label_reg(y):
    trans = y[:, 0] + np.around(y[:, 1] / 60, 2)
    return trans

y_re_train = label_reg(y_train)
y_re_valid = label_reg(y_valid)
y_re_test = label_reg(y_test)

K.clear_session()
model = regression_model()
model.fit(x=X_train, y=y_re_train, batch_size=64, epochs=30, validation_data=(X_valid, y_re_valid))


# In[4]:


tf.keras.utils.plot_model(model, to_file="./fig/reg_model.png", show_shapes=True)


# In[7]:


model.evaluate(X_test, y_re_test)


# ### Classification

# #### 24 Classes

# In[3]:


K.clear_session()
data = np.load("./a2_data/images.npy")
label = np.load("./a2_data/labels.npy")

data = data/255.


# In[33]:


y_c=[]
for i in range(24):
    la=np.zeros(24)
    la[i]=la[i]+1
    for j in range(750):
        y_c.append(la)


# In[34]:


y_c=np.stack(y_c)
y_c=np.array(y_c,dtype=np.int32)
y_c=np.concatenate((y_c,label),axis=1)
X, X_test, y, y_test = train_test_split(data, y_c, test_size=0.2, random_state=42)
y=y[:,:24]
y_test=y_test[:,24:]
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")


# In[48]:


classification_model = keras.Sequential(
    [
        keras.Input(shape=(150,150,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",strides=2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64,activation=keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(24,activation="softmax"),
    ]
)

classification_model.summary()


# In[9]:


tf.keras.utils.plot_model(classification_model, to_file="./fig/class_model.png", show_shapes=True)


# In[49]:


batch_size = 128
epochs = 40
classification_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,), metrics="accuracy", run_eagerly=True)


# In[50]:


classification_model.fit(X,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[51]:


output=classification_model.predict(X_test)


# In[52]:


def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)


# In[53]:


output=output.argmax(axis=1)
output=output*((720/24)/60)


# In[58]:


y_test=y_test[:,0]+y_test[:,1]/60


# In[60]:


common_sense_reg(y_test,output).mean()


# #### 72 Classes

# In[8]:


y_c=[]
for i in range(72):
    la=np.zeros(72)
    la[i]=la[i]+1
    for j in range(250):
        y_c.append(la)


# In[9]:


y_c=np.stack(y_c)
y_c=np.array(y_c,dtype=np.int32)
y_c=np.concatenate((y_c,label),axis=1)
X, X_test, y, y_test = train_test_split(data, y_c, test_size=0.2, random_state=42)
y=y[:,:72]
y_test=y_test[:,72:]
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")


# In[12]:


classification_model = keras.Sequential(
    [
        keras.Input(shape=(150,150,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",strides=2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64,activation=keras.layers.LeakyReLU(alpha=0.01)),
        layers.Dense(72,activation="softmax"),
    ]
)

classification_model.summary()


# In[13]:


batch_size = 128
epochs = 40
classification_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,), metrics="accuracy", run_eagerly=True)


# In[14]:


classification_model.fit(X,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[19]:


output=classification_model.predict(X_test)


# In[20]:


output=output.argmax(axis=1)


# In[21]:


output=output*((720/72)/60)


# In[22]:


def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)


# In[23]:


y_test=y_test[:,0]+y_test[:,1]/60


# In[25]:


common_sense_reg(y_test,output).mean()


# #### 720 classes

# In[7]:


y_c=[]
for i in range(720):
    la=np.zeros(720)
    la[i]=la[i]+1
    for j in range(25):
        y_c.append(la)


# In[8]:


y_c=np.stack(y_c)
y_c=np.array(y_c,dtype=np.int32)
y_c=np.concatenate((y_c,label),axis=1)
X, X_test, y, y_test = train_test_split(data, y_c, test_size=0.2, random_state=42)
y=y[:,:720]
y_test=y_test[:,720:]
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")


# In[15]:


classification_model = keras.Sequential(
    [
        keras.Input(shape=(150,150,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",strides=2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64,activation="relu"),
        layers.Dense(720,activation="softmax"),
    ]
)

classification_model.summary()


# In[16]:


batch_size = 128
epochs = 20
classification_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,), metrics="accuracy", run_eagerly=True)


# In[17]:


classification_model.fit(X,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[24]:


output=classification_model.predict(X_test)


# In[25]:


output=output.argmax(axis=1)


# In[26]:


output=output*((720/720)/60)


# In[27]:


def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)


# In[ ]:


y_test=y_test[:,0]+y_test[:,1]/60


# In[29]:


common_sense_reg(y_test,output).mean()


# #### Multi-head models

# In[7]:


X, X_test, y, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")


# In[8]:


y1=y[:,0]


# In[9]:


y2=y[:,1]


# In[10]:


y1=tf.keras.utils.to_categorical(y1)


# In[11]:


y2=y2/60


# In[12]:


y_test=y_test[:,0]+y_test[:,1]/60


# In[10]:


def multihead_model():
    in_put = layers.Input((150, 150, 1))
    x = layers.Conv2D(16, (3, 3), activation="relu")(in_put)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x1 = layers.Dense(64, activation="relu")(x)
    x1 = layers.Dense(12, activation="softmax",name="output1")(x1)
    x2 = layers.Dense(64,activation="relu")(x)
    x2 = layers.Dense(1,name="output2")(x2)
    model = tf.keras.Model(in_put, [x1,x2])
    return model


# In[11]:


multihead_model=multihead_model()


# In[20]:


multihead_model.summary()


# In[12]:


tf.keras.utils.plot_model(multihead_model, to_file="./fig/multi_model.png", show_shapes=True)


# In[21]:


batch_size = 128
epochs = 20
multihead_model.compile(loss={"output1":"categorical_crossentropy","output2":"mse"},loss_weights={"output1":1,"output2":1}, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,), metrics={"output1":"accuracy","output2":"mse"})


# In[28]:


multihead_model.fit(X,[y1,y2], batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[29]:


output1,output2=multihead_model.predict(X_test)


# In[30]:


output1=output1.argmax(axis=1)
output2=output2.reshape(-1,)


# In[31]:


output=output1+output2


# In[32]:


def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)


# In[33]:


common_sense_reg(y_test,output).mean()


# #### Label Transformation

# In[71]:


ya_hour=2*np.pi*(label[:,0]+label[:,1]/60)/12
ya_hour1=np.sin(ya_hour)
ya_hour2=np.cos(ya_hour)
y_l=np.stack((ya_hour1,ya_hour2),axis=1)
y_l=np.concatenate((y_l,label),axis=1)


# In[82]:


X, X_test, y, y_test = train_test_split(data, y_l, test_size=0.2, random_state=42)
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")
y=y[:,:2]
y_test=y_test[:,2:]


# In[107]:


transformation_model = keras.Sequential(
    [
        keras.Input(shape=(150,150,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",strides=2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64,activation="relu"),
        layers.Dense(2,activation="tanh"),
    ]
)

transformation_model.summary()


# In[13]:


tf.keras.utils.plot_model(transformation_model, to_file="./fig/label_model.png", show_shapes=True)


# In[108]:


batch_size = 128
epochs = 20
transformation_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,), metrics="mse", run_eagerly=True)


# In[109]:


transformation_model.fit(X,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[110]:


output=transformation_model.predict(X_test)


# In[111]:


predict1=output[:,0]/np.sqrt(np.square(output[:,0])+np.square(output[:,1]))


# In[112]:


predict2=output[:,1]/np.sqrt(np.square(output[:,0])+np.square(output[:,1]))


# In[134]:


import math
angle=[]
for i in range(len(predict1)):
    a_acos = math.acos(predict2[i])
    if predict1[i] < 0:
        angle.append( math.degrees(-a_acos) % 360 )
    else: 
        angle.append( math.degrees(a_acos) )


# In[135]:


angle=np.stack(angle)


# In[139]:


time_p=12*angle/360


# In[142]:


y_test=y_test[:,0]+y_test[:,1]/60


# In[143]:


def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)


# In[145]:


common_sense_reg(y_test,time_p).mean()


# #### Final Model

# In[2]:


data = np.load("./images.npy")
label = np.load("./labels.npy")
data = data / 255.


# In[3]:


ya_hour = 2 * np.pi * (label[:, 0] + label[:, 1] / 60) / 12
ya_hour1 = np.sin(ya_hour)
ya_hour2 = np.cos(ya_hour)
y_l = np.stack((ya_hour1, ya_hour2), axis=1)
y_l = np.concatenate((y_l, label), axis=1)


# In[4]:


X, X_test, y, y_test = train_test_split(data, y_l, test_size=0.2, random_state=42)
print(f"Train set: {X.shape}")
print(f"Test set: {X_test.shape}")
y = y[:, :2]
y_test = y_test[:, 2:]


# In[5]:


y_test = y_test[:, 0] + y_test[:, 1] / 60


# In[14]:


K.clear_session()
final_model = keras.Sequential(
    [
        keras.Input(shape=(150,150,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=0.01),strides=1,kernel_regularizer=regularizers.L2(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=0.01),strides=2,kernel_regularizer=regularizers.L2(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=keras.layers.LeakyReLU(alpha=0.01),strides=1,kernel_regularizer=regularizers.L2(1e-4)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(64,activation=keras.layers.LeakyReLU(alpha=0.01),kernel_regularizer=regularizers.L2(1e-4)),
        layers.Normalization(axis=None),
        layers.Dense(2,activation="tanh"),
    ]
)

final_model.summary()


# In[15]:


tf.keras.utils.plot_model(final_model, to_file="./fig/final_model.png", show_shapes=True)


# In[7]:


batch_size = 128
epochs = 300
final_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009,), metrics="mse")


# In[8]:


checkpoint_filepath = "./best_model.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mse',
    mode='min',
    save_best_only=True
)
final_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[model_checkpoint_callback])


# In[9]:


final_model.load_weights(checkpoint_filepath)
output=final_model.predict(X_test)
predict1=output[:,0]/np.sqrt(np.square(output[:,0])+np.square(output[:,1]))
predict2=output[:,1]/np.sqrt(np.square(output[:,0])+np.square(output[:,1]))


# In[10]:


angle=[]
for i in range(len(predict1)):
    a_acos = math.acos(predict2[i])
    if predict1[i] < 0:
        angle.append( math.degrees(-a_acos) % 360 )
    else: 
        angle.append( math.degrees(a_acos) )


# In[11]:


angle=np.stack(angle)
time_p=12*angle/360

def common_sense_reg(y_true, y_pred):
    d = np.abs(y_true-y_pred)
    return (1-(d//6))*(d)+(d//6)*(12-d)

common_sense_reg(y_test,time_p).mean()


# ## Task 3: Generative Models

# In this task, we use two datasets to train the generative models and obtain new figures. We explore the effects of layers and parameters in the given notebook with MNIST. After that, we leverage the power of gnerative models on Butterfly & Moth dataset.
# 1. MNIST: We directly call the Tensorflow API to download the dataset. However, the original dataset is also available on this website: <https://deepai.org/dataset/mnist>.
# 2. Butterfly & Moth: This dataset can be download from: <https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species?resource=download>.

# ### MNIST

# In[2]:


import os
import re
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, Input

path = "./archive/"
data_path = "./data/"


# In[2]:


# Download MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
X = np.concatenate((x_train, x_test), axis=0)
np.random.seed(42)
np.random.shuffle(X)
X = X.astype("float32") / 255.
X = X.reshape(X.shape + (1,))
print(f"The shape of dataset: {X.shape}")


# In[3]:


# Use modified grid_plot functions in the tutorial           
def grid_plot(images, epoch="", name="", n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()


# In[4]:


np.random.seed(42)
grid_plot(X[np.random.randint(0, 1000, 9)], name="MNIST dataset (28 X 28 X 1)", n=3)


# In[4]:


# Use modified build_conv_net and build_deconv_net functions in the tutorial
def build_conv_net(in_shape, out_shape, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    in_put = Input(shape=in_shape)
    
    x = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu')(in_put)
    x = Conv2D(kernel_size=3, filters=64, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2D(kernel_size=3, filters=64, padding='same', activation='relu')(x)
    x = Conv2D(kernel_size=3, filters=64, padding='same', activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation)(x)
    model = tf.keras.Model(in_put, x)
    model.summary()
    return model


def build_deconv_net(latent_dim, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that 
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers. 
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images. 
    """
    in_put = Input(shape=latent_dim)
    x = Dense(14 * 14 * 64)(in_put)
    x = Reshape((14, 14, 64))(x) # This matches the output size of the downsampling architecture
    x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2D(filters=1, kernel_size=3, activation=activation_out, padding='same')(x)
    model = tf.keras.Model(in_put, x)
    model.summary()
    return model


# In[5]:


conv = build_conv_net(in_shape=(28, 28, 1), out_shape=32)
tf.keras.utils.plot_model(conv, to_file="./fig/conv.png", show_shapes=True)


# In[6]:


deconv = build_deconv_net(32)
tf.keras.utils.plot_model(deconv, to_file="./fig/deconv.png", show_shapes=True)


# #### Convolutional Autoencoder (CAE)

# In[19]:


# CAE
# Use the build_convolutional_autoencoder function in the tutorial
def build_convolutional_autoencoder(data_shape, latent_dim):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid')
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    return autoencoder

# Defining the model dimensions and building it
image_size = X.shape[1:]
latent_dim = 32
cae = build_convolutional_autoencoder(image_size, latent_dim)

for epoch in range(0, 11):
    cae.fit(x=X, y=X, epochs=1, batch_size=64, verbose=0)
    print('\nEpoch: ', epoch)
    samples = X[:9]
    reconstructed = cae.predict(samples, verbose=0)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)


# #### Variational Autoencoders (VAEs)

# In[7]:


# Use the Sampling class and the build_vae function in the tutorial
class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

    
def build_vae(data_shape, latent_dim, filters=128):

    # Building the encoder - starts with a simple downsampling convolutional network  
    encoder = build_conv_net(data_shape, latent_dim*2)
    
    # Adding special sampling layer that uses the reparametrization trick 
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var])
    
    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid')
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))
    
    # Adding the special loss term
    kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
    vae.add_loss(kl_loss/tf.cast(tf.keras.backend.prod(data_shape), tf.float32))

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# In[8]:


# Training the VAE model
latent_dim = 32
encoder, decoder, vae = build_vae(X.shape[1:], latent_dim)

# Generate random vectors that we will use to sample our latent space
for epoch in range(0, 20):
    vae.fit(x=X, y=X, epochs=1, batch_size=16, verbose=1)
    latent_vectors = np.random.randn(9, latent_dim)
    images = decoder(latent_vectors)
    grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# In[32]:


latent_dim = 32
for i in range(-2, 3):
    np.random.seed(42)
    latent_vectors = np.tile(np.random.randn(latent_dim), (7, 1))
    latent_vectors[:, 9] = np.linspace(-5, 5, num=7)
    latent_vectors[:, 26] = i * np.ones(7)

    images = decoder(latent_vectors)
    plt.figure(figsize = (15, 1.5))
    for index in range(7):
        plt.subplot(1, 7, index + 1)
        plt.axis('off')
        plt.imshow(images[index], aspect='auto')
    plt.show()


# #### Generative Adversarial Networks (GANs)

# In[28]:


# Use build_gan, run_generator, get_batch, train_gan function in the tutorial
def build_gan(data_shape, latent_dim, lr=0.0002, beta_1=0.5):
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1)

    # Usually the GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh')
    
    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN = tf.keras.models.Sequential([generator, discriminator])
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator, generator, GAN


def run_generator(generator, n_samples=100):
    """
    Run the generator model and generate n samples of synthetic images using random latent vectors
    """
    latent_dim = generator.layers[0].input_shape[-1]
    generator_input = np.random.randn(n_samples, latent_dim[1])

    return generator.predict(generator_input)
    

def get_batch(generator, dataset, batch_size=64):
    """
    Gets a single batch of samples (X) and labels (y) for the training the discriminator.
    One half from the real dataset (labeled as 1s), the other created by the generator model (labeled as 0s).
    """
    batch_size //= 2 # Split evenly among fake and real samples

    fake_data = run_generator(generator, n_samples=batch_size)
    real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

    X = np.concatenate([fake_data, real_data], axis=0)
    y = np.concatenate([np.zeros([batch_size, 1]), np.ones([batch_size, 1])], axis=0)

    return X, y


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, batch_size=64):

    batches_per_epoch = int(dataset.shape[0] / batch_size / 2)
    for epoch in range(n_epochs):
        for batch in range(batches_per_epoch):
            
            # 1) Train discriminator both on real and synthesized images
            X, y = get_batch(generator, dataset, batch_size=batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)

            # 2) Train generator (note that now the label of synthetic images is reversed to 1)
            X_gan = np.random.randn(batch_size, latent_dim)
            y_gan = np.ones([batch_size, 1])
            generator_loss = gan.train_on_batch(X_gan, y_gan)
            
        noise = np.random.randn(16, latent_dim)
        images = generator.predict(noise)
        grid_plot(images, epoch, name='GAN generated images', n=3, save=False, scale=True)


# In[29]:


# Build and train the model (need around 10 epochs to start seeing some results)
latent_dim = 256
discriminator, generator, gan = build_gan(X.shape[1:], latent_dim)

train_gan(generator, discriminator, gan, X, latent_dim, n_epochs=20)


# In[66]:


latent_dim = 256

np.random.seed(13)
latent_vectors = np.tile(np.random.randn(latent_dim), (7, 1))
latent_vectors[:, 0] = np.linspace(-30, 30, num=7)

images = generator(latent_vectors)
plt.figure(figsize = (15, 1.5))
for index in range(7):
    plt.subplot(1, 7, index + 1)
    plt.axis('off')
    plt.imshow(images[index], aspect='auto')
plt.show()


# In[67]:


latent_dim = 256

np.random.seed(13)
latent_vectors = np.tile(np.random.randn(latent_dim), (7, 1))
latent_vectors[:, 119] = np.linspace(-30, 30, num=7)

images = generator(latent_vectors)
plt.figure(figsize = (15, 1.5))
for index in range(7):
    plt.subplot(1, 7, index + 1)
    plt.axis('off')
    plt.imshow(images[index], aspect='auto')
plt.show()


# In[69]:


latent_dim = 256

np.random.seed(13)
latent_vectors = np.tile(np.random.randn(latent_dim), (7, 1))
latent_vectors[:, 125] = np.linspace(0, 30, num=7)

images = generator(latent_vectors)
plt.figure(figsize = (15, 1.5))
for index in range(7):
    plt.subplot(1, 7, index + 1)
    plt.axis('off')
    plt.imshow(images[index], aspect='auto')
plt.show()


# ### Butterfly & Moth

# In[3]:


# Generate the data file
if not os.path.exists(data_path):
    os.makedirs(data_path)

count = 0
for obj in os.listdir(path):
    sub_path = os.path.join(path, obj)
    if os.path.isdir(sub_path):
        for cate in os.listdir(sub_path):
            cate_path = os.path.join(sub_path, cate)
            for file in os.listdir(cate_path):
                shutil.move(os.path.join(cate_path, file), os.path.join(data_path, f"{count}.jpg"))
                count += 1
                
shutil.rmtree(path)
print(f"Data: {count} images.")

# Convert .jpg file to numpy array and resize the image
data_lst = os.listdir(data_path)
image_arr = list()
for fig in data_lst:
    image = cv2.imread(os.path.join(data_path, fig))
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image_arr.append(np.asarray(image))
    os.remove(os.path.join(data_path, fig))
np.save(os.path.join(data_path, "part3.npy"), image_arr)


# In[3]:


# Use modified load_real_samples and grid_plot functions in the tutorial
def load_real_samples(scale=False):
    X = np.load(os.path.join(data_path, "part3.npy"))
    if scale:
        X = (X - 127.5) * 2
    return X / 255.
                
    
def grid_plot(images, epoch="", name="", n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()


# In[4]:


dataset = load_real_samples()
np.random.seed(42)
grid_plot(dataset[np.random.randint(0, 1000, 9)], name="Butterfly & Moths dataset (64 X 64 X 3)", n=3)


# In[5]:


# Use build_conv_net and build_deconv_net functions in the tutorial
def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    model = tf.keras.Sequential()
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    model.add(Conv2D(input_shape=in_shape, **default_args, filters=filters))

    for _ in range(n_downsampling_layers):
        model.add(Conv2D(**default_args, filters=filters))

    model.add(Flatten())
    model.add(Dense(out_shape, activation=out_activation) )
    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that 
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers. 
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images. 
    """
    model = tf.keras.Sequential()
    model.add(Dense(4 * 4 * 128, input_dim=latent_dim)) 
    model.add(Reshape((4, 4, 128))) # This matches the output size of the downsampling architecture
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')
    
    for i in range(n_upsampling_layers):
        model.add(Conv2DTranspose(**default_args, filters=filters))

    # This last convolutional layer converts back to 3 channel RGB image
    model.add(Conv2D(filters=3, kernel_size=(3,3), activation=activation_out, padding='same'))
    model.summary()
    return model


# In[7]:


# CAE
# Use the build_convolutional_autoencoder function in the tutorial
def build_convolutional_autoencoder(data_shape, latent_dim, filters=128):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim, filters=filters)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    return autoencoder

# Defining the model dimensions and building it
image_size = dataset.shape[1:]
latent_dim = 512
num_filters = 128
cae = build_convolutional_autoencoder(image_size, latent_dim, num_filters)

for epoch in range(0, 11):
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64, verbose=0)
    if epoch % 2 == 0:
        print('\nEpoch: ', epoch)
        samples = dataset[:9]
        reconstructed = cae.predict(samples, verbose=0)
        grid_plot(samples, epoch, name='Original', n=3, save=False)
        grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)


# In[6]:


# Use the Sampling class and the build_vae function in the tutorial
class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon

    
def build_vae(data_shape, latent_dim, filters=128):

    # Building the encoder - starts with a simple downsampling convolutional network  
    encoder = build_conv_net(data_shape, latent_dim*2, filters=filters)
    
    # Adding special sampling layer that uses the reparametrization trick 
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var])
    
    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))
    
    # Adding the special loss term
    kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
    vae.add_loss(kl_loss/tf.cast(tf.keras.backend.prod(data_shape), tf.float32))

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# In[9]:


# Training the VAE model
latent_dim = 32
encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=128)

# Generate random vectors that we will use to sample our latent space
for epoch in range(0, 101):
    vae.fit(x=dataset, y=dataset, epochs=1, batch_size=8, verbose=1)
    if epoch % 5 == 0:
        decoder.save_weights(f'./vae-models/checkpoint{epoch}')
        latent_vectors = np.random.randn(9, latent_dim)
        images = decoder(latent_vectors)
        grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# In[96]:


latent_dim = 32
_encoder, vae_decoder, _vae = build_vae(dataset.shape[1:], latent_dim, filters=128)


# In[105]:


vae_decoder.load_weights('./vae-models/checkpoint100')

for i in range(-1, 5, 2):
    np.random.seed(227)
    latent_vectors = np.tile(np.random.randn(latent_dim), (7, 1))
    latent_vectors[:, 5] = np.linspace(1.5, 7.5, num=7)
    latent_vectors[:, 28] = i * np.ones(7)

    images = vae_decoder(latent_vectors)
    plt.figure(figsize = (15, 1.5))
    for index in range(7):
        plt.subplot(1, 7, index + 1)
        plt.axis('off')
        plt.imshow(images[index], aspect='auto')
    plt.show()


# In[30]:


# Use build_gan, run_generator, get_batch, train_gan function in the tutorial
def build_gan(data_shape, latent_dim, filters=128, lr=0.0002, beta_1=0.5):
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta_1)

    # Usually the GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh', filters=filters)
    
    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1, filters=filters) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN = tf.keras.models.Sequential([generator, discriminator])
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator, generator, GAN


def run_generator(generator, n_samples=100):
    """
    Run the generator model and generate n samples of synthetic images using random latent vectors
    """
    latent_dim = generator.layers[0].input_shape[-1]
    generator_input = np.random.randn(n_samples, latent_dim)

    return generator.predict(generator_input)
    

def get_batch(generator, dataset, batch_size=64):
    """
    Gets a single batch of samples (X) and labels (y) for the training the discriminator.
    One half from the real dataset (labeled as 1s), the other created by the generator model (labeled as 0s).
    """
    batch_size //= 2 # Split evenly among fake and real samples

    fake_data = run_generator(generator, n_samples=batch_size)
    real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

    X = np.concatenate([fake_data, real_data], axis=0)
    y = np.concatenate([np.zeros([batch_size, 1]), np.ones([batch_size, 1])], axis=0)

    return X, y


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, batch_size=64):

    batches_per_epoch = int(dataset.shape[0] / batch_size / 2)
    for epoch in range(n_epochs):
        for batch in range(batches_per_epoch):
            
            # 1) Train discriminator both on real and synthesized images
            X, y = get_batch(generator, dataset, batch_size=batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)

            # 2) Train generator (note that now the label of synthetic images is reversed to 1)
            X_gan = np.random.randn(batch_size, latent_dim)
            y_gan = np.ones([batch_size, 1])
            generator_loss = gan.train_on_batch(X_gan, y_gan)
            
        if epoch % 5 == 0:
            generator.save_weights(f'./gan-models/checkpoint{epoch + 251}')
            noise = np.random.randn(16, latent_dim)
            images = generator.predict(noise)
            grid_plot(images, epoch + 251, name='GAN generated images', n=3, save=False, scale=True)


# In[31]:


# Build and train the model (need around 10 epochs to start seeing some results)
latent_dim = 256
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)
generator.load_weights('./gan-models/checkpoint250')

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=100)


# In[32]:


_discriminator, gan_generator, _gan = build_gan(dataset.shape[1:], latent_dim, filters=128)


# In[112]:


gan_generator.load_weights('./gan-models/checkpoint250')
latent_dim = 256

#1325-250, 39:0-16-14
np.random.seed(1325)
latent_vectors = np.tile(np.random.randn(latent_dim), (8, 1))
latent_vectors[:, 39] = np.linspace(0.0, 16.0, num=8)

images = gan_generator.predict(latent_vectors)
images = 0.5 * images + 0.5
plt.figure(figsize = (15, 1.5))
for index in range(8):
    plt.subplot(1, 8, index + 1)
    plt.axis('off')
    plt.imshow(images[index], aspect='auto')
plt.show()

