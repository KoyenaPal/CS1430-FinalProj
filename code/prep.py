import random
import os

import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from PIL import Image

import hyperparameters as hp




def int2bits(num):
    return np.array([int(x) for x in list(np.binary_repr(num, width=4))])

shape2vec = {
    "Circle": int2bits(0),
    "Heptagon": int2bits(1),
    "Hexagon": int2bits(2),
    "Nonagon": int2bits(3),
    "Octagon": int2bits(4),
    "Pentagon": int2bits(5),
    "Square": int2bits(6),
    "Star": int2bits(7),
    "Triangle": int2bits(8)}

def shape_embed(labels):
    return np.array([shape2vec[l] for l in labels])




def _load_image(path, prepper=None):
    img = Image.open(path).resize((hp.img_size, hp.img_size))
    img = np.array(img, dtype=np.float32) / 255.

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    return prepper(img) if prepper else img




def _standardize(img, mean, std):
    return (img - mean) / std

def _destandardize(img, mean, std):
    return (img * std) + mean




def _make_prepper(mean, std):
    def foo(img):
        return _standardize(img, mean, std)
    return foo

def _make_ender(mean, std):
    def foo(img):
        return _destandardize(img, mean, std)
    return foo




def _calc_mean_and_std(dir_path, file_ext):
    """ Calculate mean and standard deviation of a sample of the images in path_to_dir """

    # Get list of all images in training directory
    file_list = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith(file_ext):
                file_list.append(os.path.join(root, name))

    # Shuffle filepaths
    random.shuffle(file_list)

    # Take sample of file paths
    file_list = file_list[:hp.preprocess_sample_size]

    # Allocate space in memory for images
    data_sample = np.zeros(
        (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

    # Import images
    for i, file_path in enumerate(file_list):
        data_sample[i] = _load_image(file_path)

    mean = np.mean(data_sample, axis=0)
    std = np.std(data_sample, axis=0)
    return (mean, std)




def _make_shapes_batch_iters(dir_path, prepper):

    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=prepper,
        rotation_range=2.0,
        brightness_range=[0.95, 1.05],
        shear_range=5.0,
        zoom_range=[0.95, 1.05],
        fill_mode='nearest',
        horizontal_flip=True)

    filenames = []
    shapes = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.png'):
                filenames.append(os.path.join(root, name))
                shapes.append(name.partition('_')[0])

    n_files = len(filenames)
    rounded = n_files - (n_files % hp.batch_size)
    n_batches = rounded / hp.batch_size
    print(n_batches)
    n_batches = int(n_batches)

    filenames = np.array(filenames[:rounded])
    # shapes = _bert_model.encode(shapes[:rounded])
    shapes = shape_embed(shapes[:rounded])

    perm = np.random.permutation(len(shapes))
    fchunks = np.split(filenames[perm], n_batches)
    schunks= np.split(shapes[perm], n_batches)

    def xs():
        for i in range(n_batches):
            imgs = [_load_image(f, prepper) for f in fchunks[i]]
            imgs = np.array([augmentor.random_transform(img) for img in imgs])
            yield imgs, schunks[i]
    
    def ys():
        for i in range(n_batches):
            imgs = [_load_image(f, prepper) for f in fchunks[i]]
            imgs = np.array([augmentor.random_transform(img) for img in imgs])
            yield imgs

    return xs(), ys()




mean, std = _calc_mean_and_std(hp.path_to_shapes_dir, '.png')
_prepper = _make_prepper(mean, std)
ender = _make_ender(mean, std)

def save_image(img, path, ender):
    img = ender(img)
    img = img * 255.
    img = img.astype(np.uint8)
    img.save(path)

_bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
shapes_batch_gen_x, shapes_batch_gen_y = _make_shapes_batch_iters(hp.path_to_shapes_dir, _prepper)
