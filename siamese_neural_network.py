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

        :param input_image: feature vector of anchor image
        :param validation_image: feature vector of verification image
        :return: absolute distance between both feature vectors
        """
        return tf.math.abs(input_image - validation_image)


class SNN:
    """
    Building, training, saving and printing a siamese neural network architecture.
    """
    def __init__(self, x_train, x_val, y_train, y_val, epochs, lr):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.epochs = epochs
        self.lr = lr
        self._model = None

    def build_model(self):
        """
        Define model architecture.

        :return: model
        """
        input_a = Input(shape=(375, 375, 3))
        input_b = Input(shape=(375, 375, 3))

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

        feature_vector_a = snn(input_a)
        feature_vector_b = snn(input_b)
        L1Layer = SiameseL1Distance()
        similarity = L1Layer(feature_vector_a, feature_vector_b)

        output = Dense(1, activation='sigmoid')(similarity)

        self._model = Model(inputs=[input_a, input_b], outputs=output)

        return self._model

    def summary(self):
        """
        Print model architecture.

        :return: summary of model architecture
        """
        return self._model.summary()

    def train(self):
        """
        Train model architecture with model_training and validation data.
        """
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        self._model.fit(x=[self.x_train[:, 0, :, :, :], self.x_train[:, 1, :, :, :]],
                        y=self.y_train,
                        validation_data=([self.x_val[:, 0, :, :, :], self.x_val[:, 1, :, :, :]], self.y_val),
                        epochs=self.epochs,
                        verbose=1)

    def save_model(self, path):
        """
        Save trained model to path.

        :param path: path to where model shall be saved
        """
        if self._model is None:
            raise TypeError("Model is of type 'None' and must be trained before saving.")
        self._model.save(path)
