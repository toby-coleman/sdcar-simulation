from keras.models import model_from_json
import pandas as pd
from os import path
from PIL import Image
from model import preprocess_image, load_data

import matplotlib.pyplot as plt

if __name__ == '__main__':
    plot_length = 512
    # Load model and weights
    with open('model.json', 'r') as file:
        model = model_from_json(file.read())
    model.compile('adam', 'mse')
    model.load_weights('model.h5')
    # Load training data
    df = pd.read_csv('data/driving_log.csv').head(plot_length)
    gen = load_data('data', df, 128)
    # Create predictions
    predictions = model.predict_generator(gen, len(df))

    print(predictions.shape)
    # Plot to file
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df.steering, ls='--', color='b', label='actual')
    ax.plot(df.index, predictions, ls='-', color='r', label='model')
    ax.legend(loc='lower right')
    ax.set_ylabel('Steering angle')
    plt.savefig('plot.png')

