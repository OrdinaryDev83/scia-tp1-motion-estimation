import tensorflow as tf
from tensorflow.keras import layers, models, utils, applications
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# Load and explore the MNIST dataset
ds_dict, ds_info = tfds.load('mnist', as_supervised=False, with_info=True)

class_names = ds_info.features['label'].names
n_classes = len(class_names)

ds_train, ds_test = ds_dict["train"], ds_dict["test"]

# Split the data into training, validation, and testing sets

nb_examples_train = ds_info.splits["train"].num_examples
nb_examples_validation = int(nb_examples_train * 0.1)

ds_validation = ds_train.take(nb_examples_validation)
ds_train = ds_train.skip(nb_examples_validation)

def shape(x):
  return x['image'], x['label']

def preprocess(img, label):
    # img = tf.image.resize(img, [32, 32]) should be a good practice
    img = tf.cast(img, dtype=tf.float32)
    
    return img, int(label)

batch_size = 32

# applies the preprocessing functions to the images and labels, shuffles the data,
# batches it for training, and prefetches it for optimal performance
def preprocess_dataset(ds):
    ds = ds.map(shape, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(len(ds))
    # don't cache !!
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)

    return ds

ds_train, ds_validation, ds_test = preprocess_dataset(ds_train), preprocess_dataset(ds_validation), preprocess_dataset(ds_test)

# Normalize the data

def normalize(img, label):
  img = tf.cast(img / 255.0, dtype=tf.float32)
  return img, label

ds_train, ds_test = ds_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE), ds_test.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

# Build a simple feedforward neural network
model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(
    ds_train,
    batch_size=batch_size,
    epochs=5,
    validation_data=ds_validation,
    validation_batch_size=batch_size
)