import os
import random

import tensorflow as tf
import numpy as np

from prep import coco_gen, save_image, bert
from model import Holly
import hyperparameters as hp

def load_recent_priority_model():
    for root, dirs, _ in os.walk("./priority/"):
        dirs = sorted(dirs, reverse=True)
        print(dirs)

        for d in dirs:
            for root2, _, files in os.walk(os.path.join(root, d)):
                for f in files:
                    l,m,r = f.partition('.')
                    if r == 'index':
                        Holly.load_weights(os.path.join(root2, l))
                        print("Found recent priority model to evaluate!")
                        print("Loading " + root2 + "/" + l)
                        return True
                break

    return False


def load_recent_model():
    if load_recent_priority_model(): return True

    for root, _, files in os.walk("./checkpoints/"):
        files = sorted(files, reverse=True)
        print(files)
        for f in files:
            l,m,r = f.partition('.')
            if r == 'index':
                Holly.load_weights('./checkpoints/' + l)
                print("Found recent model to evaluate!")
                print("Loading " + root + l)
                return True
        break

    return False

def main():
            
    if not load_recent_model():
        printf("No model found to evaluate :(")
        return -1

    # gen = coco_gen(english=True)

    print("Saving captioned imgs to ./eval/coco_caps...")
    for x in range(10):
        img = np.random.normal(size=(1, hp.img_size, hp.img_size, 3))
        cap = bert.encode(["a black clock in the shape of a triangle"])
        cap2 = np.zeros((1, 384))
        output = Holly((img, cap)).numpy()[0]
        output2 = Holly((img, cap2)).numpy()[0]

        save_image(output, "./eval/coco_caps/"+str(x)+".jpg")
        save_image(output2, "./eval/coco_caps/"+str(x)+"-2.jpg")

if __name__ == '__main__':
        main()
