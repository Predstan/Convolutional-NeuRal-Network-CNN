#import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    #if len(sys.argv) not in [2, 3]:
        #sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data('gtsrb')

    # Split data into training and testing sets
    print(labels)
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    model.save("traffic_model")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    """train_dir = os.path.join(data_dir)
    train_label = np.array([x for x in range(NUM_CATEGORIES)])
    train_data_gen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range= 40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
    train_generator = train_data_gen.flow_from_directory(
                    train_dir,
                    target_size = (30, 30),
                    batch_size = 70,
                    class_mode = "categorical",
                    subset="training")

    test_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(30, 30),
        class_mode="categorical",
        subset="validation"
    )"""

    all_image = {}
    data = []
    labels = []
    for i in range(NUM_CATEGORIES):
        all_image[i] = glob.glob("gtsrb/"+str(i)+"/*.*")
        for key in all_image[i]:
            image=tf.keras.preprocessing.image.load_img(key, color_mode='rgb', 
            target_size= (IMG_WIDTH,IMG_HEIGHT))
            image = np.array(image)/255
            data.append(image)
            labels.append(i)
    return data, labels





def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding= 'same',input_shape = (IMG_WIDTH,IMG_HEIGHT, 3), data_format="channels_last"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(86, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




"""if __name__ == "__main__":
    main()"""

new_model = tf.keras.models.load_model('traffic_model') 
all_images = glob.glob("gtsrb/"+str(17)+"/*.*")


image=tf.keras.preprocessing.image.load_img(all_images[70], color_mode='rgb', 
        target_size= (IMG_WIDTH, IMG_HEIGHT))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.expand_dims(image, axis=0)

image = image/255
pred = new_model.predict(image)
print(np.argmax(pred))

    
