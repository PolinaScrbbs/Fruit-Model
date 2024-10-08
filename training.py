import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.regularizers import L2
import matplotlib.pyplot as plt

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./datasets/training_set",
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./datasets/training_set",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./datasets/test_set",
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
)


num_classes = len(train_ds.class_names)  # Определение количества классов
print(f"Classes: {train_ds.class_names}")  # Вывод всех классов


data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
])


model = Sequential([
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 
    data_augmentation,
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(num_classes)  
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()


EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=2,
)


epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.show()


model.evaluate(test_ds, batch_size=32, verbose=1)


model.save(f'model_epoch_{EPOCHS}.keras')
