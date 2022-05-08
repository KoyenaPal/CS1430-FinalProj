import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import hyperparameters as hp




act = tf.nn.leaky_relu
l2 = tf.keras.regularizers.L2(l2=1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate, name='Adam')





# embed_inpus = keras.Input(shape=(384,))  # We probably won't actually need 384 vals, so learn resizing to 128
# embed = layers.Dense(128, activation=act)(embed_inpus)
# embed = layers.Dense(128, activation=act)(embed)

# embed_inputs = keras.Input(shape=(4,), batch_size=hp.batch_size)
embed_inputs = keras.Input(shape=(4,))
embed = layers.Dense(128, activation=act)(embed_inputs)




# imgs_inputs = keras.Input(shape=(hp.img_size, hp.img_size, 3), batch_size=hp.batch_size)
imgs_inputs = keras.Input(shape=(hp.img_size, hp.img_size, 3))

conv = layers.Conv2D(8, 3, activation=act, kernel_regularizer=l2)(imgs_inputs)
conv = layers.LayerNormalization(axis=[1, 2])(conv)  # normalize spatially, but not across channels
conv = layers.AlphaDropout(0.05)(conv)

conv = layers.Conv2D(16, 3, activation=act, strides=2, kernel_regularizer=l2)(conv)
conv = layers.LayerNormalization(axis=[1, 2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (64, 64, 16)

conv = layers.Conv2D(32, 3, activation=act, strides=2, kernel_regularizer=l2)(conv)
conv = layers.LayerNormalization(axis=[1, 2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (32, 32, 32)

conv = layers.Conv2D(64, 3, activation=act, strides=2, kernel_regularizer=l2)(conv)
conv = layers.LayerNormalization(axis=[1, 2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (16, 16, 64)

conv = layers.Conv2D(8, 3, activation=act, strides=2, kernel_regularizer=l2)(conv)
conv = layers.LayerNormalization(axis=[1, 2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (8, 8, 8)

flat = layers.Flatten()(conv)
flat = layers.Dense(128, activation=act)(flat)

print(flat)
print(embed)

combo = layers.Concatenate()([flat, embed])
combo = layers.Dense(256, activation=act)(combo)
combo = layers.LayerNormalization()(combo)
combo = layers.Dense(128, activation=act)(combo)
combo = layers.LayerNormalization()(combo)
combo = layers.Dense(64, activation=act)(combo)
combo = layers.LayerNormalization()(combo)
combo = layers.Dense(128, activation=act)(combo)
combo = layers.LayerNormalization()(combo)
combo = layers.Dense(256, activation=act)(combo)
combo = layers.LayerNormalization()(combo)

deconv = layers.Reshape((8, 8, 4))(combo)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation=act, kernel_regularizer=l2)(deconv)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation=act, kernel_regularizer=l2)(deconv)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation=act, kernel_regularizer=l2)(deconv)
deconv = layers.Conv2DTranspose(3, 3, strides=2, activation=act, kernel_regularizer=l2)(deconv)

outputs = layers.Resizing(128, 128)(deconv)

Holly = keras.Model(inputs=(imgs_inputs, embed_inputs), outputs=outputs, name="holly")
Holly.compile(loss='mse', optimizer=optimizer)


