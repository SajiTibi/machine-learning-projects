import json

import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, concatenate, GlobalMaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def create_model():
    # Input layers
    input_1 = Input(shape=(75, 75, 3), name="bands")
    input_2 = Input(shape=(1,), name="inc_angle")
    # Hidden layers
    # layer-1
    cnn = Conv2D(32, activation='relu', kernel_size=(3, 3))(input_1)
    cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)
    # layer-2
    cnn = Conv2D(64, activation='relu', kernel_size=(3, 3))(cnn)
    cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)
    # layer-3
    cnn = Conv2D(64, activation='relu', kernel_size=(3, 3))(cnn)
    cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)
    # layer-4
    cnn = Conv2D(32, activation='relu', kernel_size=(3, 3))(cnn)
    cnn = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(cnn)
    cnn = Dropout(0.2)(cnn)
    cnn = GlobalMaxPooling2D()(cnn)
    cnn = concatenate([cnn, input_2])  # connecting angle with cnn
    # layer -5
    cnn = Dense(256, activation='relu')(cnn)
    cnn = (Dropout(0.2))(cnn)
    # layer-6
    cnn = (Dense(128, activation='relu'))(cnn)
    cnn = (Dropout(0.2))(cnn)
    # Output  layer
    cnn = (Dense(1, activation='sigmoid'))(cnn)
    model = Model(inputs=[input_1, input_2], outputs=cnn)
    model.compile(optimizer='adam', metrics=["accuracy"], loss="binary_crossentropy")
    return model


def get_data(path: str):
    data_file = open(path)
    data_json = json.load(data_file)
    band_w, band_h = 75, 75
    data_count = len(data_json)
    band_1 = np.zeros((data_count, band_w, band_h))
    band_2 = np.zeros((data_count, band_w, band_h))
    bands_avg = np.zeros((data_count, band_w, band_h))
    X_angles = np.zeros(data_count)
    Y = np.zeros(data_count)

    for i, row in enumerate(data_json):
        b1 = np.array(row["band_1"]).reshape(band_w, band_h)
        b2 = np.array(row["band_2"]).reshape(band_w, band_h)
        band_1[i] = b1
        band_2[i] = b2
        bands_avg[i] = (b1 + b2) / 2
        if str(row["inc_angle"]) == "na":
            X_angles[i] = 0
        else:
            X_angles[i] = row["inc_angle"]
        Y[i] = row["is_iceberg"]
    X_bands = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis],
                              bands_avg[:, :, :, np.newaxis]], axis=-1)
    return [X_bands, X_angles], Y


def multiple_input_generator(images_gen, X1, X2, y, batch_size):
    # https://github.com/keras-team/keras/issues/8130
    genX1 = images_gen.flow(X1, y, batch_size=batch_size, seed=7)
    genX2 = images_gen.flow(X1, X2, batch_size=batch_size, seed=7)
    while True:
        X1_i = genX1.next()
        X2_i = genX2.next()
        yield [X1_i[0], X2_i[1]], X1_i[1]


if __name__ == '__main__':
    BATCH_SIZE = 24
    EPOCHS = 20
    USE_IMAGE_AUGMENTATION = True
    [X_train_bands, X_train_angles], y = get_data("train.json/data/processed/train.json")
    model = create_model()
    X_train_bands, X_validate_bands, X_train_angles, X_validate_angles, Y_train, Y_validate = \
        train_test_split(X_train_bands, X_train_angles, y, test_size=0.20)

    if USE_IMAGE_AUGMENTATION:
        # TODO explore more about parameters
        images_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=0.1, rotation_range=10)
        # using image augmentation may lead to lower accuracy in train. however, on new data it performs with higher
        # accuracy, this because we have more images and less overfitting.
        gen = multiple_input_generator(images_gen, X_train_bands, X_train_angles, Y_train, BATCH_SIZE)
        model.fit_generator(gen, epochs=EPOCHS, verbose=2,
                            validation_data=([X_validate_bands, X_validate_angles], Y_validate),
                            steps_per_epoch=len(X_train_bands) / BATCH_SIZE)
    else:
        model.fit([X_train_bands, X_train_angles], Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=([X_validate_bands, X_validate_angles], Y_validate), verbose=2)
    score = model.evaluate([X_validate_bands, X_validate_angles], Y_validate, verbose=2)
    print(f'loss: {score[0]} accuracy: {score[1]}')
