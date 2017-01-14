import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from PIL import Image, ImageOps
from os import path
from itertools import cycle


IMAGE_SIZE = (80, 40)
ANGLE_OFFSET = 0.3


def preprocess_image(img, flip=False):
    """
    Carry out pre-processing on input images
    :param img:
    :return: normalised and re-sized image data as numpy array
    """
    # Resize image
    resized = img.resize(IMAGE_SIZE)
    # Flip, if required
    if flip:
        resized = ImageOps.mirror(resized)
    # Convert to HSV, keeping just H and S
    hsv = resized.convert(mode='HSV')
    data = np.asarray(hsv)[None, :, :, :2]
    # Normalise into range [-0.5, +0.5]
    data = data / 255. - 0.5
    return data


def round_robin(*generators):
    rr = cycle(generators)
    while True:
        yield next(next(rr))


def load_data(data_folder, df, batch_size, camera='center', offset=0.0):
    """
    Generator to produce batches of training data
    :param data_folder: folder containing driving_log.csv and the training images
    :param df: driving log dataframe
    :param batch_size: generate batches of this size
    :return: tuple of (X, y) where X is image data, y is steering angle
    """
    start_row = 0
    while True:
        # Check if we're near the end of the dataframe
        if start_row > len(df) - batch_size:
            start_row = 0
            # Shuffle dataframe
            df = df.reindex(np.random.permutation(df.index))
        # Otherwise take a batch from the dataframe
        batch_df = df.iloc[start_row:start_row + batch_size,:]
        start_row += batch_size
        # Decide whether to flip this batch
        flip = False
        # Load and pre-process each of the images in the batch
        images = [Image.open(path.join(data_folder, f.strip())) for f in batch_df[camera]]
        images = [preprocess_image(img, flip) for img in images]
        # Concantenate all images into one array
        X = np.concatenate(images)
        # Load steering angles
        y = batch_df.steering.values + offset*(-1 if flip else +1)
        yield (X, y)

def train_validate_split(data_folder, batch_size, val_fraction=0.2, camera_offset=ANGLE_OFFSET):
    """

    :param data_folder: folder containing driving_log.csv and the training images
    :param batch_size: size of each training/validation batch
    :return: (training_generator, validation_generator, n_train, n_validation)
    """
    df = pd.read_csv(path.join(data_folder, 'driving_log.csv'))
    # Shuffle and split into training and validation datasets
    df = df.reindex(np.random.permutation(df.index))
    n_train = int(len(df)*(1 - val_fraction))
    df_train = df.iloc[0:n_train - 1,:]
    df_val = df.iloc[n_train:, :]
    # Calculate usable size of each data-set allowing for complete batches
    n_train = len(df_train) - (len(df_train) % batch_size)
    n_val = len(df_val) - (len(df_val) % batch_size)
    # Interleave generators for different camera angles
    train = round_robin(load_data(data_folder, df_train, batch_size, camera='center', offset=0.0),
                        load_data(data_folder, df_train, batch_size, camera='left', offset=+camera_offset),
                        load_data(data_folder, df_train, batch_size, camera='right', offset=-camera_offset))
    val = round_robin(load_data(data_folder, df_val, batch_size, camera='center', offset=0.0),
                      load_data(data_folder, df_val, batch_size, camera='left', offset=+camera_offset),
                      load_data(data_folder, df_val, batch_size, camera='right', offset=-camera_offset))
    n_train *= 3  # x3 for each camera angle
    n_val *= 3

    return (train, val, n_train, n_val)


def create_model():
    model = Sequential()
    model.add(Convolution2D(16, 5, 5,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 5, 5,
                            border_mode='valid',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3,
                            border_mode='valid',
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(Adam(lr=0.0002), 'mse')

    return model

if __name__ == '__main__':
    model = create_model()

    gen_train, gen_val, n_train, n_val = train_validate_split('data', 128)
    print('Training epoch size: {0}, Validation epoch size: {1}'.format(n_train, n_val))

    # Save model architecture
    with open('model.json', 'w') as file:
        file.write(model.to_json())

    try:
        model.fit_generator(gen_train, n_train, 10,
                            validation_data=gen_val, nb_val_samples=n_val,
                            callbacks=[ModelCheckpoint('model.h5', save_weights_only=True)])
    except KeyboardInterrupt:
        # Allow training to be terminated early
        pass

    # Clear TF session, see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()

