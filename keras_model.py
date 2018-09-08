from keras.layers import Conv2D, MaxPool2D
from keras.layers import Input, Reshape, Embedding, Flatten, Dense, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from hyper_params import HyperParams


class KearsModel:
    def __init__(self, X, y, voca_lookup, is_embedding):
        self.X = X  # X : 2 dimensional data
        self.y = y
        self.voca_lookup = voca_lookup
        self.is_embedding = is_embedding

    def train(self):
        sequence_length = self.X.shape[1]
        vocabulary_size = len(self.voca_lookup)
        # Hyper parameter
        hparams = HyperParams.get_hyper_params()

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Input
        if self.is_embedding:
            inputs = Input(shape=(sequence_length, hparams.embedding_dim), dtype='float32')
            embedding_inputs = inputs
        else:
            inputs = Input(shape=(sequence_length,), dtype='int32')
            embedding_inputs = Embedding(input_dim=vocabulary_size,
                                         output_dim=hparams.embedding_dim,
                                         input_length=sequence_length)(inputs)
        reshape_input = Reshape((sequence_length, hparams.embedding_dim, 1))(embedding_inputs)

        # Convolution layer 1
        layer_0 = Conv2D(hparams.num_filters,
                         kernel_size=(hparams.filter_sizes[0], hparams.embedding_dim),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu')(reshape_input)
        layer_0 = MaxPool2D(pool_size=(sequence_length - hparams.filter_sizes[0] + 1, 1),
                            strides=(1, 1),
                            padding='valid')(layer_0)

        # Convolution layer 2
        layer_1 = Conv2D(hparams.num_filters,
                         kernel_size=(hparams.filter_sizes[1], hparams.embedding_dim),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu')(reshape_input)
        layer_1 = MaxPool2D(pool_size=(sequence_length - hparams.filter_sizes[1] + 1, 1),
                            strides=(1, 1),
                            padding='valid')(layer_1)

        # Convolution layer 3
        layer_2 = Conv2D(hparams.num_filters,
                         kernel_size=(hparams.filter_sizes[2], hparams.embedding_dim),
                         padding='valid',
                         kernel_initializer='normal',
                         activation='relu')(reshape_input)
        layer_2 = MaxPool2D(pool_size=(sequence_length - hparams.filter_sizes[2] + 1, 1),
                            strides=(1, 1),
                            padding='valid')(layer_2)

        concatenated_tensor = Concatenate(axis=1)([layer_0, layer_1, layer_2])

        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(hparams.drop_prob)(flatten)

        # Output layers
        outputs = Dense(units=hparams.dim_output,
                        activation='softmax')(dropout)

        # Checkpoint callback
        checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')

        optimizer = Adam(lr=hparams.learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)

        # Model creation
        model = Model(inputs, outputs)
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train,
                  batch_size=hparams.batch_size,
                  epochs=hparams.epochs,
                  verbose=1,
                  callbacks=[checkpoint],
                  validation_data=(X_test, y_test))

        # Avoid the NoneType error after training.
        K.clear_session()
