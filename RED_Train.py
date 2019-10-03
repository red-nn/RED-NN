from __future__ import print_function
import os
from layer_definition import *
import tensorflow as tf

# Don't pre-allocate all GPU memory; allocate only as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ----------------------------------------------------------------------------------------------------------------------
# Load the 60,000 MNIST up-right oriented training samples and transform labels to one out of many format
data = np.load(f'MNIST_UR_train.npz')
x_train, y_train = data['x'], tf.keras.utils.to_categorical(data['y'], 10)

# Load the 10,000 MNIST randomly rotated training samples and transform labels to one out of many format
data = np.load('MNIST_RR_test.npz')
x_test, y_test = data['x'], tf.keras.utils.to_categorical(data['y'], 10)
# ----------------------------------------------------------------------------------------------
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
# ----------------------------------------------------------------------------------------------------------------------

# Input image size
ROWS = 28
COLS = 28

# Angular samples (PHI)
PHI = 16

# Number of filters for each convolution stage
CONV1_F = 16
CONV2_F = 16
CONV3_F = 16

# ----------------------------------------------------------------------------------------------------------------------
# Model definition
in_shape = tf.keras.Input(shape=(ROWS, COLS, 1), name="Input")

# Roto-translational feature space generation
x = Rig2DConv(filters=1, kernel_size=10, phi=PHI, un_rotate=True, name="Rig")(in_shape)
# Periodic padding for cyclic convolution
x = Periodic_Pad()(x)

# 3 stacked convolutional predictors with batch normalization
x = tf.keras.layers.BatchNormalization(name="BN1")(x)
x = tf.keras.layers.Conv3D(CONV1_F, kernel_size=(1, 5, 5), activation='relu', name="CONV1")(x)
x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), name="MP1")(x)

x = tf.keras.layers.BatchNormalization(name="BN2")(x)
x = tf.keras.layers.Conv3D(CONV2_F, kernel_size=(1, 3, 3), activation='relu', name="CONV2")(x)
x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), name="MP2")(x)

# Translational convolution over the depth axis of the feature space
x = tf.keras.layers.BatchNormalization(name="BN3")(x)
x = tf.keras.layers.Conv3D(CONV3_F, kernel_size=(PHI, 3, 3), activation='relu', name="CONV3")(x)

# Dense hidden layer predictor
x = tf.keras.layers.Reshape((PHI, 3*3*CONV3_F))(x)
x = tf.keras.layers.Dense(30, activation='relu', name="Hidden_Dense")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(10, activation='softmax', name="Output_table")(x)

# Maxpooling applied to the columns results in the predicted class
number = tf.keras.layers.GlobalMaxPooling1D(name="number")(x)
x_permuted = tf.keras.layers.Permute((2, 1))(x)
# Maxpooling applied to the rows results in the predicted angle row index
angle = tf.keras.layers.GlobalMaxPooling1D(name="angle")(x_permuted)
# ----------------------------------------------------------------------------------------------------------------------

model = tf.keras.Model(inputs=in_shape, outputs=number)
print(model.summary())

# ----------------------------------------------------------------------------------------------------------------------
# Training section
opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

name = f'URT_REDNN_{PHI}'

# LR decay when there is no improvement on 5 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
checkpoint = tf.keras.callbacks.ModelCheckpoint(name + '.h5', verbose=1, save_best_only=True, monitor='val_loss', mode='min')
tensorboardcb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('tests', name), write_images=True, write_grads=True)

model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[reduce_lr, checkpoint, tensorboardcb])
# ----------------------------------------------------------------------------------------------------------------------
