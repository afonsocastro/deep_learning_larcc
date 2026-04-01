#!/usr/bin/env python3

from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical  # one-hot encode target column
import numpy as np
from src.create_models import create_cnn_v1_3, create_lstm_v1_5, create_transformer_v1_1
from config.definitions import ROOT_DIR


if __name__ == '__main__':

    time_steps = 20
    # time_steps = 50
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)
    validation_split = 0.3

    training_data = np.load(ROOT_DIR + "/haptic_data/data1/200ms/normalized_train_data_200ms.npy")
    x_train = np.reshape(training_data[:, :-1], (training_data.shape[0], time_steps, 13))
    y_train = to_categorical(training_data[:, -1])

    x_train = x_train[:, :, 1:]
    print("x_train.shape")
    print(x_train.shape)
    print("y_train.shape")
    print(y_train.shape)

    model, model_name = create_transformer_v1_1()

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file=model_name+".png", show_shapes=True, show_layer_names=True, rankdir="TB", dpi=200)

    # callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)]
    fit_history = model.fit(x_train, y_train, shuffle=True, validation_split=validation_split, epochs=500, batch_size=32)
                            # callbacks=callbacks)
    model.save(model_name+".keras")
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(fit_history.history['accuracy'])
    plt.plot(fit_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #
    plt.show()