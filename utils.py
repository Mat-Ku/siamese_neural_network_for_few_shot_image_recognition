import itertools
import numpy as np
import os
import tensorflow as tf


def get_file_names(data_path):
    """
    Retrieve file names of image files and save them in a dictionary.

    :param data_path: path to data set
    :return: dictionary containing file names
    """

    images = {'damaged': {'side': [],
                          'top': []},
              'intact': {'side': [],
                         'top': []}
              }

    for path, directories, files in os.walk(data_path):
        for file in files:
            f = os.path.join(path, file).replace('\\', '/')
            if '/damaged' in f:
                if '/side' in f:
                    images['damaged']['side'].append(f)
                else:
                    images['damaged']['top'].append(f)
            else:
                if '/side' in f:
                    images['intact']['side'].append(f)
                else:
                    images['intact']['top'].append(f)

    return images


def load_image(file_path):
    """
    Load, resize and rescale image and return it as a tensor.

    :param file_path: path to image file
    :return: image as tensor
    """

    # Read in image from file path as byte file
    byte_img = tf.io.read_file(file_path)

    # Load byte image as tensor
    img = tf.io.decode_png(contents=byte_img, channels=3)

    # Resize image to square shape since original shape is (540, 960, 3)
    img = tf.image.resize(images=img, size=(375, 375))

    # Rescale pixel values to range between 0 and 1
    img = img / 255.0

    return img


def make_paired_dataset(set_a, set_b, y_set_b):
    """
    Create a data set of two image data sets by applying a cartesian product.

    :param set_a: anchor image data set
    :param set_b: image data set of positive or negative class
    :param y_set_b: class labels of set_B
    :return: image pairs, labels
    """

    # Load images as tensors
    imgs_a = [load_image(path) for path in set_a]
    imgs_b = [load_image(path) for path in set_b]

    # Cartesian product between both image sets
    x_pairs, y = [], []
    for t in itertools.product(imgs_a, imgs_b):
        img_a = t[0]
        img_b = t[1]

        x_pairs.append([img_a, img_b])
        y.append(y_set_b)

    x_pairs = np.array(x_pairs)
    y = np.array(y)

    return x_pairs, y
