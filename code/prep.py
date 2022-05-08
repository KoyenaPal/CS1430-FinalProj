import random

from sentence_transformers import SentenceTransformer

import hyperparameters as hp




_bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
shapes_batch_gen = make_shapes_batch_generator(hp.path_to_shapes_dir)




def _load_image(path, prepper=None):
    img = Image.open(f).resize(hp.img_size, hp.img_size)
    img = np.array(img, dtype=np.float32) / 255.

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    return prepper(img) if prepper else img




def make_shapes_batch_generator(dir_path):
    mean, std = _calc_mean_and_std(hp.path_to_img_dir)
    prepper = _make_prepper(mean, std)

    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=prepper,
        rotation_range=2.0,
        brightness_range=[0.95, 1.05],
        shear_range=5.0,
        zoom_range=[0.95, 1.05],
        fill_mode='nearest',
        horizontal_flip=True)

    filenames = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.png'):
                file_list.append(os.path.join(root, name))

    n_files = len(file_list)
    rounded = n_files - (n_files % hp.batch_size)
    n_batches = rounded / hp.batch_size

    filenames = file_list[0:rounded]
    shapes = _bert_model.encode([s.partition('_') for s in file_list])

    filenames_batched = np.split(np.array(file_list), n_batches).tolist()
    shapes_batched = np.split(np.array(file_list), n_batches).tolist()

    batches = list(zip(files_batched, shapes_batched))
    random.shuffle(batches)

    while batches:
        filenames, shapes = batches.pop()

        imgs = [_load_image(f, prepper) for f in names]
        imgs = [augmentor.random_transform(img) for img in imgs]

        yield (imgs, shapes)




def _calc_mean_and_std(dir_path, file_ext)
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




def _standardize(img, mean, std):
    return (img - mean) / std




def _make_standardizer(mean, std):
    def foo(img):
        return _standardize(img, mean, std)

    return foo
