from prep import shapes_batch_gen
from utils import CustomModelSaver
from model import Holly

def main():
    # load, prep, and batch image data along with label data
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(hp.checkpoint_path, hp.max_num_weights)
    ]

    model.fit(
        x=shapes_batch_gen,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list)
    )

if __name__ == '__main__':
        main()

# steps: pipeline for loading image data
# pipeline for loading text data
# transform text data with huggingface
# architecture to pass data through
