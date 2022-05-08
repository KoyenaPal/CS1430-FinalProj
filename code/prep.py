from sentence_transformers import SentenceTransformer

import hyperparameters as hp

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

mean, std = calc_mean_and_std(hp.path_to_img_dir)
prepper = make_prepper(mean, std)
train_data, test_data = get_data(hp.path_to_img_dir, True, True, prepper)

def shapes_batch_generator(dir_path):
    file_list = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.png'):
                file_list.append(os.path.join(root, name))

    n_files = len(file_list)
    rounded = n_files - (n_files % hp.batch_size)
    file_list = file_list[0:rounded]
    random.shuffle(file_list)




# def coco_batch_generator(dir_path, captions_path):


def calc_mean_and_std(dir_path, file_ext)
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
        img = Image.open(file_path)
        img = img.resize((hp.img_size, hp.img_size))
        img = np.array(img, dtype=np.float32)
        img /= 255.

        # Grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        data_sample[i] = img

    mean = np.mean(data_sample, axis=0)
    std = np.std(data_sample, axis=0)
    return (mean, std)

def standardize(img, mean, std):
    return (img - mean) / std

def make_prepper(mean, std):
    def preprocess_fn(img):
        img = img / 255.
        return self.standardize(img)

    return preproces_fn

def get_data(dir_path, shuffle, augment, prepper):
    """Returns a (train/test) pair of iterable image-batch generators, shuffle/augment are bools, prepper is preprocess function"""

    if augment:
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=prepper,
            rotation_range=2.0,
            brightness_range=[0.95, 1.05],
            shear_range=5.0,
            zoom_range=[0.95, 1.05],
            fill_mode='nearest',
            horizontal_flip=True
            )

    else:
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=prepper)

    train_gen = data_gen.flow_from_directory(
        dir_path,
        target_size=(hp.img_size, hp.img_size),
        color_mode='rgb',
        class_mode='input',
        batch_size=hp.batch_size,
        shuffle=shuffle,
        seed=0,
        save_to_dir='./augments/',
        save_format='jpg',
        subset='training')

    test_gen = data_gen.flow_from_directory(
        dir_path,
        target_size=(hp.img_size, hp.img_size),
        color_mode='rgb',
        class_mode='input',
        batch_size=hp.batch_size,
        shuffle=shuffle,
        seed=0,
        save_to_dir='./augments/',
        save_format='jpg',
        subset='testing')

    return (train_gen, test_gen)


def get_coco():
    """Returns (train_imgs, val_imgs, train_captions, val_captions) for coco dataset"""
    mean, std = calc_mean_and_std(hp.path_to_coco_img_dir)
    prepper = make_prepper(mean, std)
    train_data, val_data = get_data(hp.path_to_coco_img_dir, True, True, prepper)

def get_shapes():
    """Returns (train_imgs, val_imgs, train_captions, val_captions) for coco dataset"""
    mean, std = calc_mean_and_std(hp.path_to_shapes_dir)
    prepper = make_prepper(mean, std)
    
    
