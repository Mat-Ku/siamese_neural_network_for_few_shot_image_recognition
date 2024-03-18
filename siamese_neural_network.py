import tensorflow as tf
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import Model, Sequential


class SiameseL1Distance(Layer):
    """
    L1 distance layer designed as custom keras layer.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_image, validation_image):
        """
        L1 distance computation between convolutional feature representation of two images.

        :param input_image: Anchor image
        :param validation_image: Verification image
        :return: Absolute distance
        """
        return tf.math.abs(input_image - validation_image)


class SNN:
    """
    Building, printing and training a siamese neural network architecture.
    """

    def __init__(self, X_train, X_val, y_train, y_val, epochs, lr):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.epochs = epochs
        self.lr = lr
        self._model = None

    def build_model(self):
        """
        Define model architecture.

        :return: model variable
        """
        input_A = Input(shape=(375, 375, 3))
        input_B = Input(shape=(375, 375, 3))

        snn = Sequential([
            Conv2D(filters=32, kernel_size=(10, 10), activation='relu'),
            MaxPooling2D(pool_size=(4, 4), strides=2, padding='same'),

            Conv2D(filters=32, kernel_size=(7, 7), activation='relu'),
            MaxPooling2D(pool_size=(4, 4), strides=2, padding='same'),

            Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
            MaxPooling2D(pool_size=(4, 4), strides=2, padding='same'),

            Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
            Flatten(),
            Dense(1024, activation='sigmoid')
        ])

        feature_vector_A = snn(input_A)
        feature_vector_B = snn(input_B)
        L1Layer = SiameseL1Distance()
        similarity = L1Layer(feature_vector_A, feature_vector_B)

        output = Dense(1, activation='sigmoid')(similarity)

        self._model = Model(inputs=[input_A, input_B], outputs=output)

        return self._model

    def summary(self):
        """
        Print model architecture.

        :return: Summary of model variable
        """
        return self._model.summary()

    def train(self):
        """
        Train model architecture with training and validation data.

        """
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        self._model.fit(x=[self.X_train[:, 0, :, :, :], self.X_train[:, 1, :, :, :]],
                        y=self.y_train,
                        validation_data=([self.X_val[:, 0, :, :, :], self.X_val[:, 1, :, :, :]], self.y_val),
                        epochs=self.epochs,
                        verbose=1)
