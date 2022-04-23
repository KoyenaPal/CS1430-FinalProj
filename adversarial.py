# Adversarial discriminator against model.py

class Discriminator(tf.keras.Model):
        def __init__(self, vocab_size, window_size=10, img_width=256):

                ######vvv DO NOT CHANGE vvv##################
                super(Holly, self).__init__()

                self.vocab_size = vocab_size # The size of the english vocab used in labels
                self.window_size = window_size # Window size for attention head
                ######^^^ DO NOT CHANGE ^^^##################


                # Define batch size and optimizer/learning rate
                self.batch_size = 128
                self.embedding_size = 256
                self.learning_rate = 2e-3
                self.stddev = 1e-2

                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

                # Define english and french embedding layers:
                self.fr_embedding = tf.Variable(tf.random.truncated_normal(
                    [self.french_vocab_size, self.embedding_size], stddev=self.stddev))

                self.en_embedding = tf.Variable(tf.random.truncated_normal(
                    [self.english_vocab_size, self.embedding_size], stddev=self.stddev))

                # Create positional encoder layers
                self.fr_position = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
                self.en_position = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)

                # Define encoder and decoder layers:
                self.encoder = transformer.Transformer_Block(self.embedding_size, False)
                self.decoder = transformer.Transformer_Block(self.embedding_size, True)

                # Seems like embedding size is pretty constant. That's why we need this final dense layer, to take it to vocab size.
                # It's weird, but you CAN do all the steps of the decoder at once. It's just, each word of the output (there will be window_size words)
                # will be dependent only on the correct english words before it. So it's like if we iterated up to do it.
        
                # Define dense layer(s)
                self.dense = tf.keras.layers.Dense(self.english_vocab_size)

        @tf.function
        def call(self, encoder_input, decoder_input):
                """
                :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
                """
        
                # TODO:
                #1) Add the positional embeddings to french sentence embeddings
                #2) Pass the french sentence embeddings to the encoder
                #3) Add positional embeddings to the english sentence embeddings
                #4) Pass the english embeddings and output of your encoder, to the decoder
                #5) Apply dense layer(s) to the decoder out to generate probabilities

                fr_embeds = self.fr_position.call(tf.nn.embedding_lookup(self.fr_embedding, encoder_input))
                encoded = self.encoder.call(fr_embeds)

                en_embeds = self.en_position.call(tf.nn.embedding_lookup(self.en_embedding, decoder_input))
                decoded = self.decoder.call(en_embeds, context=encoded)
                logits = self.dense(decoded)

                return tf.nn.softmax(logits)

        def eval(self, encoder_input):
            returns sequence of word IDs, obtained by argmaxing logits



        def accuracy_function(self, prbs, labels, mask):
                """
                DO NOT CHANGE

                Computes the batch accuracy
                
                :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
                :param labels:  integer tensor, word prediction labels [batch_size x window_size]
                :param mask:  tensor that acts as a padding mask [batch_size x window_size]
                :return: scalar tensor of accuracy of the batch between 0 and 1
                """

                decoded_symbols = tf.argmax(input=prbs, axis=2)
                accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
                return accuracy


        def loss_function(self, prbs, labels, mask):
                """
                Calculates the model cross-entropy loss after one forward pass
                Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

                :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
                :param labels:  integer tensor, word prediction labels [batch_size x window_size]
                :param mask:  tensor that acts as a padding mask [batch_size x window_size]
                :return: the loss of the model as a tensor
                """

                # Note: you can reuse this from rnn_model.

                prbs = tf.boolean_mask(prbs, mask)
                labels = tf.boolean_mask(labels, mask)
                return tf.math.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

        @av.call_func
        def __call__(self, *args, **kwargs):
                return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
