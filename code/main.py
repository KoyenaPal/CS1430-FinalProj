import tensorflow as tf

from prep import shapes_batch_gen_x, shapes_batch_gen_y, save_image
from utils import CustomModelSaver
from model import Holly
import hyperparameters as hp


def main():
    # load, prep, and batch image data along with label data
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=hp.logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(hp.checkpoint_path, hp.max_num_weights)
    ]

    b = next(shapes_batch_gen_x)
    print(b[0][0].shape)
    print(b[1].shape)
    # inputs = b[0]
    # labs = b[1]
    # print(len(inputs))
    # print(len(labs))

    # i0 = inputs[0]
    # print(len(i0))
    # print(i0[0].shape)

    Holly.train_on_batch(next(shapes_batch_gen_x), y=next(shapes_batch_gen_y))

if __name__ == '__main__':
        main()
