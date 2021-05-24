import tensorflow as tf
import json
from model import NTMModel
from test_data_generator import generate_test_data_tsp


class SequenceCrossEntropyLoss(tf.keras.losses.Loss):
    eps = 1e-8

    def call(self, y_true, y_pred):
        return -tf.reduce_mean(  # cross entropy function
            y_true * tf.math.log(y_pred + self.eps) + (1 - y_true) * tf.math.log(1 - y_pred + self.eps)
        )


def train(config):
    model = NTMModel(
        batch_size=config['batch_size'],
        vector_dim=config['vector_dim'],
        model_type=config['model_type'],
        cell_params=config['cell_params'][config['model_type']]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer)
    sequence_loss_func = SequenceCrossEntropyLoss()
    save_path = ''
    for batch_index in range(config['num_batches']):
        seq_length = config['max_seq_length']
        x, y_true = generate_test_data_tsp(config['batch_size'], seq_length)
        with tf.GradientTape() as tape:
            y_pred = model((x, seq_length))
            loss = sequence_loss_func(y_true, y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 1000 == 0:
            x,y_true = generate_test_data_tsp(config['batch_size'], seq_length)
            y_pred = model((x, seq_length))
            loss = sequence_loss_func(y_true, y_pred)
            print("batch %d: loss %f" % (batch_index, loss))
            # print("original string sample: ", x[0])
            # print("copied string sample: ", y_pred[0])
            checkpoint = tf.train.Checkpoint(model)
            save_path = checkpoint.save('./checkpoints/training_checkpoints')

    print('training complete')


def test(config):
    model = NTMModel(
        batch_size=config['batch_size'],
        vector_dim=config['vector_dim'],
        model_type=config['model_type'],
        cell_params=config['cell_params'][config['model_type']]
    )
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore('./checkpoints/training_checkpoints-1').expect_partial()

    x_test, y_test = generate_test_data_tsp(config['batch_size'], config['max_seq_length'])
    y_pred = model((x_test, config['max_seq_length']))

    sequence_loss_func = SequenceCrossEntropyLoss()
    loss = sequence_loss_func(y_true=x_test, y_pred=y_pred)
    print('loss: ', loss)
    print(' x: ', y_test[0])
    print(' y: ', y_pred[0])


if __name__ == '__main__':
    with open("config.json") as f:
        config = json.load(f)
    print(config)
    if config['mode'] == 'train':
        train(config)
    else:
        test(config)
