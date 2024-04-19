from keras.models import load_model

from siamese_neural_network import SiameseL1Distance
from utils import get_file_names, make_paired_dataset

import numpy as np


def main():

    snn = load_model(filepath='model_training/trained_snn.h5',
                     custom_objects={'SiameseL1Distance': SiameseL1Distance})

    # Load data consisting of an anchor class, a positive class (both intact packages) and a
    # negative class (damaged packages)
    data_path = '*'  # insert path to data set
    img_dict = get_file_names(data_path)
    imgs_anchor = img_dict['intact']['side']
    imgs_pos = img_dict['intact']['side']
    imgs_neg = img_dict['damaged']['side']

    # Create 32 paired image samples, of which 16 are positive samples and 16 are negative samples
    pos_samples, pos_labels = make_paired_dataset(imgs_anchor[:4], imgs_pos[4:8], 1)  # yields 16
    neg_samples, neg_labels = make_paired_dataset(imgs_anchor[:4], imgs_neg[:4], 0)  # yields 16

    # The last seven samples per class are used as test data, since the other samples were used for training/validation
    x_test = np.vstack((pos_samples[9:], neg_samples[9:]))
    y_test = np.concatenate((pos_labels[9:], neg_labels[9:]), axis=None)

    # Predict test data
    # The tensors of both images classes are passed as a list with two elements to the predict() method
    y_pred = [0 if p < 0.5 else 1 for p in snn.predict([x_test[:, 0, :, :, :], x_test[:, 1, :, :, :]])]
    acc = (list(np.array(y_test - y_pred)).count(0) / len(y_test)) * 100
    print('Accuracy: {:.2f}%'.format(acc))


if __name__ == '__main__':
    main()
