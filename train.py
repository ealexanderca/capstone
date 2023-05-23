import os.path

import tensorflow as tf
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt

from data_interface.data_processor import DataProcessor
from data_interface.training_data_cache import TrainingDataCache
from data_interface.training_data_fuzzer import TrainingDataFuzzer
from data_interface.raw_data_sqlite import RawDataSQLite

USE_GPU = False

if USE_GPU:
    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise Exception("GPU not found")
else:
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
    # tf.debugging.set_log_device_placement(True)

assert tf.executing_eagerly()

tf.random.set_seed(77997468)

BATCH_SIZE = 100  # How many series are trained together
SHUFFLE_BUFFER_SIZE = 4096  # Buffer size for shuffle operation
VALIDATE_BATCHES = 10000 // BATCH_SIZE  # How many batches to reserve for validation
LOSS_THRESHOLD = -np.inf  # Threshold at which to determine that training has been successful
MAX_EPOCHS = 500  # Maximum number of epochs to train on before giving up

MIN_PROBABILITY = 1E-8  # Minimum probability for maximum likelihood estimation
MIN_SCALE = 0  # Minimum value for weibull scale
MIN_SHAPE = 0  # Minimum value for weibull scale

VALIDATION_DATA_PATH = './cache/validation_data.npz'
MODEL_PATH = './cache/model'
MODEL_CHECKPOINT_PATH = './cache/model_checkpoint'


class end_cb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < LOSS_THRESHOLD):
            print("\nLoss meets success threshold")
            self.model.stop_training = True

        if not tf.math.is_finite(logs.get('loss')):
            self.model.stop_training = True


def weibull_pdf(scale, shape, t):
    pdf = tf.math.divide_no_nan(shape, scale) * tf.math.divide_no_nan(t, scale) ** (shape - 1) * tf.math.exp(
        -tf.math.divide_no_nan(t, scale) ** shape)
    return pdf


def weibull_cdf(scale, shape, t):
    cdf = 1 - tf.math.exp(-tf.math.divide_no_nan(t, scale) ** shape)
    return cdf


def weibull_mean(scale, shape):
    # TODO: Should use gamma function from TF if possible
    return scale * scipy.special.gamma(1 + 1 / shape)

def weibull_mode(scale, shape):
    return scale*((shape-1)/shape)**(1/shape)

def weibull_median(scale, shape):
    return scale * tf.math.log(2.0) ** (1/shape)

def weibull_inv_cdf(scale, shape, cum_prob):
    return scale*(-tf.math.log(1-cum_prob))**(1/shape)


def weibull_neg_log_likelihood_loss(y_true, y_pred):
    weibull_scale = y_pred[:, 0] + MIN_SCALE
    weibull_shape = y_pred[:, 1] + MIN_SHAPE
    t_real = y_true[:, 0]

    probabilities = weibull_pdf(weibull_scale, weibull_shape, t_real)
    probabilities += MIN_PROBABILITY  # Helps prevent NaNs if guessed distribution is to far/tight. Known method used in cross-entropy loss as well

    log_likelihood = tf.math.log(probabilities)

    is_nan = tf.math.is_nan(log_likelihood)
    log_likelihood = tf.where(is_nan, tf.math.log(MIN_PROBABILITY), log_likelihood)

    log_likelihood = tf.math.reduce_sum(log_likelihood)

    loss = -log_likelihood / BATCH_SIZE

    if not tf.math.is_finite(loss):
        loss = tf.abs(loss)  # Ensure it is positive so that checkpoint manager doesn't interpret as improvement

    return loss


def cached_processor(name, test_num, start_time):
    return TrainingDataCache(name + '_processor',
                             lambda: DataProcessor(
                                 RawDataSQLite(
                                     name + '.db',
                                     test_num),
                                 start_time
                             )
                             )


def cached_training_data(name, test_num, start_time, num):
    return TrainingDataCache(name + '_fuzzer',
                             lambda: TrainingDataFuzzer(cached_processor(name, test_num, start_time), num))


def main():
    data = [
        cached_training_data('Test_Mar18_Paint_CompressorB_100Duty_Paint_from11_49_CapB', 1, 704, 30000),
        #cached_training_data('Test_Mar18_Paint_CompressorC_100Duty_Paint_from11_00_CapB', 1, 662),  # ENCODER DOES NOT WORK
        cached_training_data('Test_Mar23_Paint_CompressorA_50Duty_Paint_from12_12_CapB', 2, 752, 30000),
        cached_training_data('Test_Mar23_Paint_CompressorA_100Duty_Paint_from11_00_CapB', 1, 695, 30000),
        cached_training_data('Test_Mar28_Paint_CompressorB_50Duty_Paint_from11_40_CapA', 1, 673, 30000),
        #cached_training_data('Test_Mar28_Paint_CompressorB_50Duty_Paint_from11_15_CapA_Validation_StartingPressure_26psi', 1, 672),
    ]

    val_idx = 0

    #data_val = data[val_idx]
    data_val = data.pop(val_idx)


    data_vars = []
    data_labels = []
    data_source_idx = []
    for i, d in enumerate(data):
        data_vars.append(d.variables)
        data_labels.append(d.labels)
        idx = np.zeros(np.shape(d.labels)[0], dtype=int) + i
        data_source_idx.append(idx)

    data_vars = np.stack(data_vars)
    data_vars = data_vars.reshape((-1, *data_vars.shape[2:]), order='F')
    data_vars = data_vars.astype('float32')

    data_labels = np.stack(data_labels)
    data_labels = data_labels.reshape((-1, *data_labels.shape[2:]), order='F')
    data_labels = data_labels.astype('float32')

    data_source_idx = np.stack(data_source_idx)
    data_source_idx = data_source_idx.reshape((-1, *data_source_idx.shape[2:]), order='F')

    print(f"Data shape: {data_vars.shape}")
    print(f"Labels shape: {data_labels.shape}")
    print(f"Total sequences: {data_vars.shape[0]}")
    print(f"Variables min/max: {np.min(data_vars)}, {np.max(data_vars)}")

    ds = tf.data.Dataset.from_tensor_slices((data_vars, data_labels, data_source_idx))
    ds = ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(
        BATCH_SIZE)  # All shapes are: (batch, time, features) https://www.tensorflow.org/tutorials/structured_data/time_series

    train_ds = ds.skip(VALIDATE_BATCHES)

    #val_ds = ds.take(VALIDATE_BATCHES)
    val_vars = data_val.variables.astype('float32')
    val_labels = data_val.labels.astype('float32')
    val_ds = tf.data.Dataset.from_tensor_slices((val_vars, val_labels, np.zeros(val_labels.shape[0], dtype=int)))
    val_ds = val_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).take(VALIDATE_BATCHES)

    num_train_batches = len(list(train_ds))
    num_val_batches = len(list(val_ds))
    print(f"Training batches: {num_train_batches}")
    print(f"Validation batches: {num_val_batches}")

    train_source_idx = train_ds.map(lambda *x: x[2])
    train_ds = train_ds.map(lambda *x: x[:2])

    val_source_idx = val_ds.map(lambda *x: x[2])
    val_ds = val_ds.map(lambda *x: x[:2])

    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(8, kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2,
                              activation='relu',
                              bias_initializer=tf.constant_initializer(1),
                              kernel_regularizer=tf.keras.regularizers.l2(0.0),
                              kernel_constraint=tf.keras.constraints.non_neg(),
                              bias_constraint=tf.keras.constraints.non_neg()
                              )
        # Initializing the bias (especially for the weibull scale output) ensures that the likelihood estimate is non-zero and the gradient is measurable
        # relu output ensures that weibull parameters are always positive to prevent NaN
    ])

    steps = [10 * num_train_batches, 25 * num_train_batches, 200 * num_train_batches]
    learning_rates = [20E-4, 10E-4, 5E-4, 2E-4]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        steps, learning_rates)

    model.compile(loss=weibull_neg_log_likelihood_loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
                  # If learning rate is too fast, NaNs can occur
                  metrics=[weibull_neg_log_likelihood_loss]
                  )

    model.build(train_ds.element_spec[0].shape)

    model.summary()

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_PATH, save_best_only=True, save_weights_only=False)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)

    history = model.fit(train_ds, epochs=MAX_EPOCHS,
                        validation_data=val_ds,
                        callbacks=[end_cb(), checkpoint_best, early_stopping])

    val_loss, val_weibull = model.evaluate(val_ds)

    print('Val Loss:', val_loss)
    print('Val Weibull:', val_weibull)

    for i, d in enumerate(data):
        d_vars = d.variables.astype('float32')
        d_labels = d.labels.astype('float32')
        d_ds = tf.data.Dataset.from_tensor_slices((d_vars, d_labels))
        d_ds = d_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).take(VALIDATE_BATCHES)

        d_loss, d_weibull = model.evaluate(d_ds)

        print(f"Training dataset #{i}: Loss = {d_loss}")

    print(f"Abs Mean Weights = {[np.mean(np.abs(w)) for w in model.weights]}")
    print(f"Abs Median Weights = {[np.median(np.abs(w)) for w in model.weights]}")
    print(f"Abs Max Weights = {[np.max(np.abs(w)) for w in model.weights]}")

    print("Done.")

    model.save(MODEL_PATH)

    x_real, y_real = zip(*[x for x in val_ds.as_numpy_iterator()])
    x_real = np.concatenate(x_real)
    y_real = np.concatenate(y_real)
    source_idx = np.concatenate(list(val_source_idx))

    np.savez(VALIDATION_DATA_PATH, x_real, y_real, source_idx)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
