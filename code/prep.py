import random
import os
import json


import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from PIL import Image

import hyperparameters as hp




_bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bert = _bert_model




def int2bits(num):
    bits = [int(x) for x in list(np.binary_repr(num, width=4))]
    rv = bits
    return np.array(rv).flatten()


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

# print(shape2vec['Triangle'])
# print(shape2vec['Star'])

def shape_embed(labels):
    return np.array([shape2vec[l] for l in labels])




def _load_image(path, augment=False):
    img = Image.open(path).resize((hp.img_size, hp.img_size))
    img = (np.array(img, dtype=np.float32) / 127.5) - 1.

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    if augment:
        augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=2.0,
            brightness_range=[0.95, 1.05],
            shear_range=5.0,
            zoom_range=[0.95, 1.05],
            fill_mode='nearest',
            horizontal_flip=True)

        img = augmentor.random_transform(img)

    return img





def save_shapes_image(img, path):
    img = (img + 1.) * 127.5
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

# def save_coco_image(img, path):
#     img = _coco_ender(img)
#     img = img * 255.
#     # img = np.clip(img, 0, 255)
#     img = img.astype(np.uint8)
#     img = Image.fromarray(img)
#     img.save(path)




# def _standardize(img, mean, std):
#     return img - 0.5
#     # return (img - mean) / std

# def _destandardize(img, mean, std):
#     return img + 0.5
#     # return (img * std) + mean




# def _make_prepper(mean, std):
#     def foo(img):
#         return _standardize(img, mean, std)
#     return foo

# def _make_ender(mean, std):
#     def foo(img):
#         return _destandardize(img, mean, std)
#     return foo




# def _calc_mean_and_std(dir_path, file_ext):
#     """ Calculate mean and standard deviation of a sample of the images in path_to_dir """

#     # Get list of all images in training directory
#     file_list = []
#     for root, _, files in os.walk(dir_path):
#         for name in files:
#             if name.endswith(file_ext):
#                 file_list.append(os.path.join(root, name))

#     # Shuffle filepaths
#     random.shuffle(file_list)

#     # Take sample of file paths
#     file_list = file_list[:hp.preprocess_sample_size]

#     # Allocate space in memory for images
#     data_sample = np.zeros(
#         (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

#     # Import images
#     for i, file_path in enumerate(file_list):
#         data_sample[i] = _load_image(file_path)

#     mean = np.mean(data_sample, axis=0)
#     std = np.std(data_sample, axis=0)
#     return (mean, std)




def _make_shapes_batch_iters(dir_path, english=False):

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
    n_batches = int(n_batches)

    filenames = np.array(filenames[:rounded])
    # shapes = _bert_model.encode(shapes[:rounded])
    shapes = shapes[:rounded]

    if not english:
        shapes = shape_embed(shapes)

    pairs = list(zip(filenames, shapes))
    np.random.shuffle(pairs)
    filenames, shapes = list(zip(*pairs))
        
    fchunks = np.split(np.array(filenames), n_batches)
    schunks= np.split(np.array(shapes), n_batches)

    for i in range(n_batches):
        imgs = np.array([_load_image(f) for f in fchunks[i]])
        yield (imgs, schunks[i]), imgs




def _make_coco_batch_iters(dir_path, anno_path, prepper, english=False):

    filenames = {}
    descs = {}

    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.jpg'):
                img_id = int(name.partition('.')[0])
                filenames[img_id] = os.path.join(root, name)
                descs[img_id] = []

    anno_file = open(anno_path)
    data = json.load(anno_file)
    img_info = data['annotations']

    info_len = len(img_info)
    print("loading descriptions... [", end="", flush=True)
    for i, d in enumerate(img_info):
        embed = d['caption']
        descs[d['image_id']].append(embed)

        if (i+1) % (hp.batch_size*8) == 0:
            print("-", end="", flush=True)
    print("]")

    pairs = []

    for img_id, fname in filenames.items():
        for desc in descs[img_id]:
            pairs.append((fname, desc))

    random.shuffle(pairs)
        
    n_pairs = len(pairs)
    rounded = n_pairs - (n_pairs % hp.batch_size)
    n_batches = rounded / hp.batch_size
    n_batches = int(n_batches)

    pairs = pairs[:rounded]
    filenames, descs = zip(*pairs)
        
    fchunks = np.split(np.array(filenames), n_batches)
    dchunks = np.split(np.array(descs), n_batches)

    for i in range(n_batches):
        imgs = np.array([_load_image(f) for f in fchunks[i]])
        # vchunks = bert.encode(dchunks[i])
        vchunks = np.zeros((hp.batch_size, 384))
        yield (imgs, vchunks), imgs




# shapes_mean, shapes_std = _calc_mean_and_std(hp.path_to_shapes_dir, '.png')
# _shapes_prepper = _make_prepper(shapes_mean, shapes_std)
# _shapes_ender = _make_ender(shapes_mean, shapes_std)

# coco_mean, coco_std = _calc_mean_and_std(hp.path_to_coco_img_dir, '.jpg')
# _coco_prepper = _make_prepper(coco_mean, coco_std)
# _coco_ender = _make_ender(coco_mean, coco_std)



def shapes_gen(english=False):
    """Returns a fresh iterator over all shapes batches each call"""
    return _make_shapes_batch_iters(hp.path_to_shapes_dir, english)

def coco_gen(english=False):
    """Returns a fresh iterator over all coco batches each call"""
    return _make_coco_batch_iters(hp.path_to_coco_img_dir, hp.path_to_coco_annos, english)
