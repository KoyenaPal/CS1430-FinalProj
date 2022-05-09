import random
from datetime import datetime, date
import os

import tensorflow as tf

from prep import shapes_gen, save_shapes_image, save_coco_image, shape_embed
from utils import CustomModelSaver
from model import Holly
import hyperparameters as hp

def my_loss_fn(y_true, y_pred):
    # squared_difference = tf.square(tf.square(y_true - y_pred))
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=[1,2])

def main():
    
    for root, _, files in os.walk("./checkpoints/"):
        files = sorted(files, reverse=True)
        print(files)
        for f in files:
            l,m,r = f.partition('.')
            if r == 'index':
                Holly.load_weights('./checkpoints/' + l)
                print("Found model to continue training!")
                break
        break
            

    # train_loss = my_loss_fn
    train_loss = tf.keras.losses.MeanSquaredError()
    train_metric = tf.keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)

    Holly.compile(loss=train_loss, metrics=[train_metric], optimizer=optimizer)

    Holly.summary();


    for epoch in range(hp.num_epochs):
        gen = shapes_gen()

        for batch, (x, y) in enumerate(gen):
            Holly.train_on_batch(x, y)
            tf.summary.scalar('loss', train_metric.result(), step=batch)

        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, train_metric.result()))

    gen = shapes_gen(english=True)

    print("Saving model weights...")
    now = datetime.now()
    Holly.save_weights('./checkpoints/weights__' + date.today().isoformat() + "__" + "{}-{}".format(str(now.hour).zfill(2), str(now.minute).zfill(2)))

    print("Saving generated image selection...")
    for batch, ((imgs, shapes), _) in enumerate(gen):
        ind = random.randrange(hp.batch_size)
        img = imgs[ind:ind+1]
        shape = shapes[ind]
        shapevec = shape_embed([shape])
        output = Holly((img, shapevec)).numpy()[0]
        # print(output.shape)

        save_shapes_image(output, "./augments/"+shape+str(batch)+".png")

        


if __name__ == '__main__':
        main()
