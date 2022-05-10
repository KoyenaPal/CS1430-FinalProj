import random
import os
import json
from collections import defaultdict


import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from PIL import Image

import hyperparameters as hp




# _bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# bert = _bert_model




def int2bits(num):
    bits = [int(x) for x in list(np.binary_repr(num, width=4))]
    rv = bits
    return np.array(rv).flatten()


# Simple embedding for use on shapes task
shape2vec = {
    # "Circle": int2bits(0),
    "Heptagon": int2bits(1),
    "Hexagon": int2bits(2),
    # "Nonagon": int2bits(3),
    # "Octagon": int2bits(4),
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





def save_image(img, path):
    img = (img + 1.) * 127.5
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)




# shapes dataset doesn't really require this kind of standardization
def _standardize(img, mean, std):
    return img - 0.5
    # return (img - mean) / std

def _destandardize(img, mean, std):
    return img + 0.5
    # return (img * std) + mean




def _make_prepper(mean, std):
    def foo(img):
        return _standardize(img, mean, std)
    return foo

def _make_ender(mean, std):
    def foo(img):
        return _destandardize(img, mean, std)
    return foo



# adapted from homework 5, although variance in estimated mean was causing issues
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


def removesuffix(s, sfx):
    return s.partition(sfx)[0]

# def _make_shapes_batch_iters(dir_path, english=False):
#     filenames = []
#     shapes = []
#     for root, _, files in os.walk(dir_path):
#         for name in files:
#             if name.endswith('.png'):
#                 filenames.append(os.path.join(root, name))
#                 shapes.append(removesuffix(name.partition('_')[2], ".png"))

#     n_files = len(filenames)
#     rounded = n_files - (n_files % hp.batch_size)
#     n_batches = rounded / hp.batch_size
#     n_batches = int(n_batches)

#     filenames = np.array(filenames[:rounded])
#     # shapes = _bert_model.encode(shapes[:rounded])
#     shapes = shapes[:rounded]

#     if not english:
#         shapes = shape_embed(shapes)

#     pairs = list(zip(filenames, shapes))
#     np.random.shuffle(pairs)
#     filenames, shapes = list(zip(*pairs))
        
#     fchunks = np.split(np.array(filenames), n_batches)
#     schunks= np.split(np.array(shapes), n_batches)

#     for i in range(n_batches):
#         imgs = np.array([_load_image(f) for f in fchunks[i]])
#         yield (imgs, schunks[i]), imgs

# Custom dataset generator compatible with keras.Model.fit
# Modified from above to achieve shape-independent latent image
def _make_shapes_batch_iters(dir_path, english=False):
    d_filenames = defaultdict(lambda: [])
    for root, _, files in os.walk(dir_path):
        files.sort()

        for name in files:
            if name.endswith('.png'):
                fname = os.path.join(root, name)
                rname = name.partition(".")[0]
                uid, _, shape = rname.partition("_")
                d_filenames[uid].append((fname, shape, fname))
        break

    for k in d_filenames:
        (f0, s0, _), (f1, s1, _) = d_filenames[k]
        d_filenames[k].append((f0, s1, f1))
        d_filenames[k].append((f1, s0, f0))

    trips = list(d_filenames.values())
    trips = sum(trips, [])

    n_trips = len(trips)
    rounded = n_trips - (n_trips % hp.batch_size)
    n_batches = rounded // hp.batch_size
    n_batches = int(n_batches)

    trips = trips[:rounded]
    random.shuffle(trips)

    fname_ins, shapes, fname_outs = zip(*trips)
    if not english:
        shapes = shape_embed(shapes)

    fin_chunks = np.split(np.array(fname_ins), n_batches)
    s_chunks = np.split(np.array(shapes), n_batches)
    fout_chunks = np.split(np.array(fname_outs), n_batches)

    for i in range(n_batches):
        imgs_in = np.array([_load_image(f) for f in fin_chunks[i]])
        imgs_out = np.array([_load_image(f) for f in fout_chunks[i]])
        yield ((imgs_in, s_chunks[i]), imgs_out)



def _make_coco_pairs(dir_path, anno_path, english=False):
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

    n_pairs = len(pairs)
    rounded = n_pairs - (n_pairs % hp.batch_size)
    n_batches = rounded / hp.batch_size
    n_batches = int(n_batches)

    pairs = pairs[:rounded]
    return pairs

def _make_coco_batch_iters(pairs):
    assert(len(pairs) % hp.batch_size == 0)
    n_batches = len(pairs) // hp.batch_size
    filenames, descs = zip(*pairs)
        
    fchunks = np.split(np.array(filenames), n_batches)
    dchunks = np.split(np.array(descs), n_batches)

    for i in range(n_batches):
        imgs = np.array([_load_image(f) for f in fchunks[i]])
        vchunks = bert.encode(dchunks[i])
        yield (imgs, vchunks), imgs


# coco_pairs = _make_coco_pairs(hp.path_to_coco_img_dir, hp.path_to_coco_annos, False)
# coco_pairs_eng = _make_coco_pairs(hp.path_to_coco_img_dir, hp.path_to_coco_annos, True)

def shapes_gen(english=False):
    """Returns a fresh iterator over all shapes batches each call"""
    return _make_shapes_batch_iters(hp.path_to_shapes_dir, english)

def coco_gen(english=False):
    """Returns a fresh iterator over all coco batches each call"""
    random.shuffle(coco_pairs)
    if english:
        return _make_coco_batch_iters(coco_pairs_eng)
    return _make_coco_batch_iters(coco_pairs)
