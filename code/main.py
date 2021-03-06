import random
from datetime import datetime, date, timezone, timedelta
import time
import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from prep import shapes_gen, coco_gen, save_shapes_image, shape_embed, bert
from utils import CustomModelSaver
from model import Holly
import hyperparameters as hp


def my_loss_fn(y_true, y_pred):
    # squared_difference = tf.square(tf.square(y_true - y_pred))
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=[1,2])

def sec2hms(sec):
    sec = int(sec)
    sec_h = sec//3600
    sec -= sec_h * 3600
    sec_m = sec//60
    sec -= sec_m * 60

    return str(sec_h).zfill(2), str(sec_m).zfill(2), str(sec).zfill(2)

def save_weights(model):
    now = datetime.now(timezone(timedelta(hours=-4)))
    filename = './checkpoints/weights__' + date.today().isoformat() + "__" + "{}-{}".format(str(now.hour).zfill(2), str(now.minute).zfill(2))
    print("Saving model weights as " + filename)
    model.save_weights(filename)
        


def main():

    # for root, _, files in os.walk("./checkpoints/"):
    #     files = sorted(files, reverse=True)
    #     print(files)
    #     for f in files:
    #         l,m,r = f.partition('.')
    #         if r == 'index':
    #             Holly.load_weights('./checkpoints/' + l)
    #             print("Found model to continue training!")
    #             break
    #     break
            

    # train_loss = my_loss_fn
    train_loss = tf.keras.losses.MeanSquaredError()
    train_metric = tf.keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)

    Holly.compile(loss=train_loss, metrics=[train_metric], optimizer=optimizer)

    Holly.summary();


    
    for epoch in range(hp.num_epochs):
        gen = shapes_gen()
        start = time.time()

        for batch, (x, y) in enumerate(gen):
            Holly.train_on_batch(x, y)

        end = time.time()
        diff = end-start
        exptr = diff * (hp.num_epochs - (epoch + 1))
        diff_s = "{}:{}:{}".format(*sec2hms(diff))
        exptr_s = "{}:{}:{}".format(*sec2hms(exptr))
        template = 'Epoch {}, \tLoss: {}, \tTime: {}, \tExpected time remaining: {}'
        print (template.format(epoch+1, train_metric.result(), diff_s, exptr_s))

        if epoch % 100 == 0:
            save_weights(Holly)

    save_weights(Holly)



    gen = shapes_gen(english=True)

    print("Saving generated image selection...")
    for batch, ((imgs, shapes), _) in enumerate(gen):
        ind = random.randrange(hp.batch_size)
        img = imgs[ind:ind+1]
        shape = shapes[ind]
        shapevec = shape_embed([shape])
        output = Holly((img, shapevec)).numpy()[0]

        save_shapes_image(output, "./augments/"+shape+str(batch)+"-generated.png")
        save_shapes_image(img[0], "./augments/"+shape+str(batch)+"-input.png")

def coco_main():

    for root, _, files in os.walk("./checkpoints/"):
        files = sorted(files, reverse=True)
        print(files)
        for f in files:
            l,m,r = f.partition('.')
            if r == 'index':
                Holly.load_weights('./checkpoints/' + l)
                print("Found model to continue training!")
                print("Loading " + root + "/" + l)
                break
        break
            

    train_loss = tf.keras.losses.MeanSquaredError()
    train_metric = tf.keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)

    Holly.compile(loss=train_loss, metrics=[train_metric], optimizer=optimizer)
    Holly.summary();

    for epoch in range(hp.num_epochs):
        gen = coco_gen()
        start = time.time()

        for batch, (x, y) in enumerate(gen):
            Holly.train_on_batch(x, y)

        end = time.time()
        diff = end-start
        exptr = diff * (hp.num_epochs - (epoch + 1))
        diff_s = "{}:{}:{}".format(*sec2hms(diff))
        exptr_s = "{}:{}:{}".format(*sec2hms(exptr))
        template = 'Epoch {}: \tLoss: {}, \tTime: {}, \tExpected time remaining: {}'
        print (template.format(epoch+1, train_metric.result(), diff_s, exptr_s))

        if (epoch+1) % 4 == 0:
            save_weights(Holly)

    save_weights(Holly)
    gen = coco_gen(english=True)

    print("Saving generated image selection...")
    for batch, ((imgs, descs), _) in enumerate(gen):
        ind = random.randrange(hp.batch_size)
        img = imgs[ind:ind+1]
        desc = descs[ind]
        descvec = bert.encode([desc])
        output = Holly((img, descvec)).numpy()[0]

        save_shapes_image(output, "./augments/"+desc[0:10]+str(batch)+"-generated.png")
        save_shapes_image(img[0], "./augments/"+desc[0:10]+str(batch)+"-input.png")

        


if __name__ == '__main__':
        coco_main()
