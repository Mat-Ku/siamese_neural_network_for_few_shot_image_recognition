from siamese_neural_network import SNN
from utils import get_file_names, make_paired_dataset

import numpy as np


def main():

    # Load data
    data_path = '*'  # insert path to data set
    img_dict = get_file_names(data_path)
    imgs_anchor = img_dict['intact']['side']
    imgs_pos = img_dict['intact']['side']
    imgs_neg = img_dict['damaged']['side']

    # Create 32 paired image samples, of which 16 are positive samples and 16 are negative samples
    pos_samples, pos_labels = make_paired_dataset(imgs_anchor[:4], imgs_pos[4:8], 1)  # yields 16
    neg_samples, neg_labels = make_paired_dataset(imgs_anchor[:4], imgs_neg[:4], 0)  # yields 16

    # Compose model_training data of two positive samples and two negative samples
    x_train = np.vstack((pos_samples[:2], neg_samples[:2]))
    y_train = np.concatenate((pos_labels[:2], neg_labels[:2]), axis=None)

    # Compose validation data of seven positive samples and seven negative samples
    # The remaining seven samples in each class are preserved for the test data
    x_val = np.vstack((pos_samples[2:9], neg_samples[2:9]))
    y_val = np.concatenate((pos_labels[2:9], neg_labels[2:9]), axis=None)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # Train model
    snn = SNN(x_train=x_train,
              x_val=x_val,
              y_train=y_train,
              y_val=y_val,
              epochs=10,
              lr=0.001)
    snn.build_model()
    snn.summary()
    snn.train()

    # Save model
    #snn.save_model('trained_snn.h5')  


if __name__ == '__main__':
    main()
