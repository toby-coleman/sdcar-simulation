from keras.models import model_from_json
from keras import backend as K

import pandas as pd
import numpy as np
from model import load_data

import matplotlib.pyplot as plt

if __name__ == '__main__':
    plot_length = 1000
    # Load model and weights
    with open('model.json', 'r') as file:
        model = model_from_json(file.read())
    model.compile('adam', 'mse')
    model.load_weights('model.h5')
    # Load training data
    df = pd.read_csv('data/driving_log.csv')
    gen = load_data('data', df, 128, camera='center')
    # Create predictions
    predictions = model.predict_generator(gen, len(df) - len(df) % 128)
    print('Mean angle: ', np.mean(predictions))

    K.clear_session()

    # Plot to file
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index[:plot_length], df.steering.iloc[:plot_length], ls='--', color='b', label='actual')
    ax.plot(df.index[:plot_length], predictions[:plot_length], ls='-', color='r', label='model')
    ax.grid()
    ax.set_axis_bgcolor('linen')
    ax.legend(loc='lower right')
    ax.set_xlim([0, plot_length])
    ax.set_ylabel('Steering angle')
    plt.savefig('plot.png')

