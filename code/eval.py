import os
import random

import tensorflow as tf

from prep import shapes_gen, save_shapes_image, save_coco_image, shape_embed
from model import Holly
import hyperparameters as hp

def load_recent_priority_model():
    for root, dirs, _ in os.walk("./priority/"):
        dirs = sorted(dirs)
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

    gen = shapes_gen(english=True)

    print("Saving generated image selection to ./eval...")
    for batch, ((imgs, shapes), _) in enumerate(gen):
        ind = random.randrange(hp.batch_size)
        img = imgs[ind:ind+1]
        shape = shapes[ind]
        shapevec = shape_embed([shape])
        output = Holly((img, shapevec)).numpy()[0]

        save_shapes_image(output, "./eval/"+shape+str(batch)+"-generated.png")
        save_shapes_image(img[0], "./eval/"+shape+str(batch)+"-input.png")

if __name__ == '__main__':
        main()
