from __future__ import print_function
import itertools
import time
import numpy as np
from matplotlib import pyplot as plt
from layer_definition import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scipy.ndimage import rotate

# Don't pre-allocate all GPU memory; allocate only as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Confusion matrix plot function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix (rotMNIST)',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix1.png", dpi=300)


# --------------------------------------------------------------------------------------------------------------------
# Load ROT validation set
data = np.load('MNIST_RR_test.npz')
x_test, y_test = data['x'], tf.keras.utils.to_categorical(data['y'], 10)

# --------------------------------------------------------------------------------------------------------------------
x_test = np.reshape(x_test, (-1, 28, 28, 1))
x_test = x_test / 255.0
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# Load model
model = tf.keras.models.load_model('RRT_REDNN_16.h5', custom_objects={'Rig2DConv': Rig2DConv,
                                                                              'Periodic_Pad': Periodic_Pad})
# --------------------------------------------------------------------------------------------------------------------

# Print summary and learned weights
print(model.summary())
weights = model.layers[1].get_weights()
print(f'Trained parameters: l={weights[0]}, alpha={weights[1]}, beta={weights[2]}')

# Print accuracy and loss
t1 = time.time()
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test time:', time.time()-t1)

# --------------------------------------------------------------------------------------------------------------------
# Plot confusion matrix
Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true = np.argmax(y_test, axis = 1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(10), normalize=False)
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# Print table P and angular indexes
for _ in range(8, -8, -1):
    input_image = x_test[1051]
    input_image = rotate(input_image, 22.50 * _, reshape=False)
    input_image = np.reshape(input_image, (1, 28, 28, 1))

    # Input image plot
    # plt.imshow(input_image[0, :, :, 0])
    # plt.show()

    layer_name = 'Output_table'
    int_output = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    intout = int_output.predict(input_image)
    out_row = np.unravel_index(intout.argmax(), intout.shape)

    # Print all the table
    # print(np.round(intout, 2))

    # Print the predicted index row
    print(f"Predicted index row: {out_row[1]}")
# --------------------------------------------------------------------------------------------------------------------
