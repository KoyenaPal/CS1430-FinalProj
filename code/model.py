import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import hyperparameters as hp

def bias_l2():
    return tf.keras.regularizers.l2(l2=1e-3)

def kernel_l2():
    return tf.keras.regularizers.l2(l2=1e-3)

# avoid dropout at transition points, and on embeddings, since those seem less likely to be "smooth"?
# avoid activation near end, since colors aren't binary




########## Image convolution (shrinking)
imgs_inputs = keras.Input(shape=(hp.img_size, hp.img_size, 3))

conv = layers.Conv2D(8, 3, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(imgs_inputs)
conv = layers.LayerNormalization(axis=[1,2])(conv)
# (126, 126, 8)

conv = layers.Conv2D(16, 3, activation="leaky_relu", strides=2, kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(conv)
conv = layers.LayerNormalization(axis=[1,2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (62, 62, 16)

conv = layers.Conv2D(32, 3, activation="leaky_relu", strides=2, kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(conv)
conv = layers.LayerNormalization(axis=[1,2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (30, 30, 32)

conv = layers.Conv2D(64, 3, activation="leaky_relu", strides=2, kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(conv)
conv = layers.LayerNormalization(axis=[1,2])(conv)
conv = layers.AlphaDropout(0.05)(conv)
# (14, 14, 64)

conv = layers.Conv2D(16, 3, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(conv)
conv = layers.LayerNormalization(axis=[1,2])(conv)
# (12, 12, 16)




########## Image flattening, final dense processing before concatenation
flat = layers.Flatten()(conv)
flat = layers.Dense(768, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(flat)
flat = layers.LayerNormalization(axis=[1])(flat)




########## Sentence embedding
embed_inputs = keras.Input(shape=(4,))
embed = layers.Dense(384, activation="leaky_relu", bias_initializer='glorot_uniform')(embed_inputs)
embed = layers.LayerNormalization(axis=[1])(embed)
embed = layers.Dense(384, activation="leaky_relu", bias_initializer='glorot_uniform')(embed)
embed = layers.LayerNormalization(axis=[1])(embed)




########## Bottleneck (concatenating sentence embedding + latent image)
########## Later layers of the bottleneck get residuals of initial concatenation, so they don't have to remember semantics

concat = layers.Concatenate()([flat, embed])
combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(concat)
combo = layers.LayerNormalization(axis=[1])(combo)
combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(layers.Concatenate()([combo, concat]))
combo = layers.LayerNormalization(axis=[1])(combo)
combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(layers.Concatenate()([combo, concat]))
combo = layers.LayerNormalization(axis=[1])(combo)
combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(layers.Concatenate()([combo, concat]))
combo = layers.LayerNormalization(axis=[1])(combo)

# combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(flat)
# combo = layers.LayerNormalization(axis=[1])(combo)
# combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(combo)
# combo = layers.LayerNormalization(axis=[1])(combo)
# combo = layers.Dense(54, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(combo)
# combo = layers.LayerNormalization(axis=[1])(combo)
# combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(layers.Concatenate()([combo, embed]))
# combo = layers.LayerNormalization(axis=[1])(combo)
# combo = layers.Dense(1024, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(layers.Concatenate()([combo, embed]))
# combo = layers.LayerNormalization(axis=[1])(combo)




########## Deconvolution (growing)
########## No activation on final 2 layers since values of color channels are evenly distributed
deconv = layers.Reshape((8, 8, 16))(combo)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(deconv)
deconv = layers.LayerNormalization(axis=[1, 2])(deconv)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(deconv)
deconv = layers.LayerNormalization(axis=[1, 2])(deconv)
deconv = layers.Conv2DTranspose(16, 3, strides=2, activation="leaky_relu", kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(deconv)
deconv = layers.LayerNormalization(axis=[1, 2])(deconv)
deconv = layers.Conv2DTranspose(16, 3, strides=2, kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(deconv)
deconv = layers.LayerNormalization(axis=[1, 2])(deconv)
deconv = layers.Conv2D(3, 16, kernel_regularizer=kernel_l2(), bias_regularizer=bias_l2(), bias_initializer='glorot_uniform')(deconv)

outputs = deconv
print(outputs)  # shape sanity check

Holly = keras.Model(inputs=(imgs_inputs, embed_inputs), outputs=outputs, name="holly")
Holly_alt = keras.Model(inputs=(flat, embed_inputs), outputs=outputs, name="holly2")
