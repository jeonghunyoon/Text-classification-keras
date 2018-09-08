import tensorflow as tf


class HyperParams:
    @staticmethod
    def get_hyper_params():
        return tf.contrib.training.HParams(
            embedding_dim=256,
            filter_sizes=[3, 4, 5],
            num_filters=512,
            drop_prob=0.5,

            epochs=100,
            batch_size=30,
            dim_output=2,
            learning_rate=1e-4
        )
