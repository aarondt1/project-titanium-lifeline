import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('INFO')
import logging

tf.get_logger().setLevel(logging.ERROR)
import keras
import keras_applications
from metrics import auc_roc
from keras import backend as K
from keras.models import load_model
from keras_contrib.applications.nasnet import NASNetLarge, NASNET_LARGE_WEIGHT_PATH_WITH_auxiliary
from keras.utils import multi_gpu_model

MODEL_CHECKPOINT = "./output/model_durchlauf2_LARGE.15-0.44.h5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
K.tensorflow_backend.set_session(tf.Session(config=config))

df_train = pd.read_csv('CheXpert-v1.0-small/train.csv').fillna(0)  # [:500]
# converting -1 to random interval
ran = pd.DataFrame(np.random.uniform(0.55, 0.85, size=(len(df_train.index), len(df_train.columns))),
                   columns=df_train.columns, index=df_train.index)
df_train = df_train.where(df_train != -1.0, ran)

# no post-processing necessary due to the intended lack of uncertainty labels / nans
df_val = pd.read_csv('CheXpert-v1.0-small/valid.csv')  # [:500]


class CondBCE(keras.losses.Loss):
    def call(self, y_true, y_pred):  # (bs, classes)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.mean(K.binary_crossentropy(y_true[:, :-1], y_pred[:, :-1], from_logits=False) * K.round(y_true[:, -1, None]), axis=-1)


def data_generators(batch_size, img_dim):
    img_data_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=7,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1 / 255)

    # GENIUS
    def generator_wrapper(gen):
        def cond(labels, indicies, mother, neg):
            siblings = labels[:, [*indicies, *mother]]
            siblings[:, -len(mother):] = np.where(neg, np.nan_to_num(siblings[:, -len(mother):])
                                                  < 0.5, np.isfinite(siblings[:, -len(mother):])).astype(float)
            # return siblings
            return np.hstack(
                [siblings[:, :-len(mother)], np.product(siblings[:, -len(mother):], axis=1, keepdims=True)])

        for iter_gen in gen:
            labels = iter_gen[1]

            head = cond(labels, [0, 13], [True], [False])
            top = cond(labels, [1, 3, 4, 9 or 10 or 11, 12], [0], [True])
            cardio = cond(labels, [2], [0, 1], [True, False])
            lung = cond(labels, [4, 5, 6, 7, 8], [0, 3], [True, False])
            pleural = cond(labels, [9, 10, 11], [0], [True])

            yield iter_gen[0], [iter_gen[1], head, top, cardio, lung, pleural]

    train_gen = img_data_gen.flow_from_dataframe(df_train,
                                                 directory=None,
                                                 x_col='Path',
                                                 y_col=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                                                        'Lung Opacity',
                                                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                                                        'Atelectasis',
                                                        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                                                        'Support Devices'],
                                                 target_size=img_dim,
                                                 color_mode='grayscale',
                                                 class_mode='raw',
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 interpolation='box')
    val_gen = img_data_gen.flow_from_dataframe(df_val,
                                               directory=None,
                                               x_col='Path',
                                               y_col=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                                                      'Lung Opacity',
                                                      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                                                      'Atelectasis',
                                                      'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                                                      'Support Devices'],
                                               target_size=img_dim,
                                               color_mode='grayscale',
                                               class_mode='raw',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               interpolation='box')

    return generator_wrapper(train_gen), generator_wrapper(val_gen), len(train_gen), len(val_gen)


def create_model(img_dim):
    # backbone = keras.applications.nasnet.NASNetLarge(input_shape=(*img_dim, 1), include_top=False, weights=None,
    backbone = NASNetLarge(input_shape=(*img_dim, 1),
                           dropout=0.5,
                           weight_decay=5e-5,
                           use_auxiliary_branch=True,
                           include_top=True,
                           weights=None,
                           input_tensor=None,
                           pooling=None,
                           classes=14,
                           activation='sigmoid')

    weights_path = keras.utils.get_file(
        'nasnet_large_with_aux.h5',
        NASNET_LARGE_WEIGHT_PATH_WITH_auxiliary,
        cache_subdir='models')
    backbone.load_weights(weights_path, by_name=True, skip_mismatch=True)
    # backbone.load_weights(MODEL_CHECKPOINT, by_name=True, skip_mismatch=True)
    return classifier(backbone)


def classifier(model, weight_decay=5e-5):
    x = model.get_layer('dropout_1').output
    head = keras.layers.Dense(3, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='sigmoid', name='head')(x)
    top = keras.layers.Dense(6, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='sigmoid', name='top')(x)
    cardio = keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='sigmoid', name='cardio')(x)
    lung = keras.layers.Dense(6, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='sigmoid', name='lung')(x)
    pleural = keras.layers.Dense(4, kernel_regularizer=keras.regularizers.l2(weight_decay), activation='sigmoid', name='pleural')(x)

    clsfr = keras.Model(model.input, [model.output[1], head, top, cardio, lung, pleural])
    return clsfr

def accuracy(y_true, y_pred):
    y_true = y_true[:, :-1]
    y_pred = y_pred[:, :-1]
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.cast(K.equal(y_true, y_pred), K.floatx())

# set initial_epoch to last successful epoch
def train(model, epochs, train_gen, val_gen, train_size, val_size, initial_epoch=0):
    filepath = './output/model_cond_LARGE.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    logdir = "./logs/scalars/"  # /scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tensorboard_callback.samples_seen = initial_epoch  # * len(train_gen)
    tensorboard_callback.samples_seen_at_last_write = tensorboard_callback.samples_seen

    model.compile(keras.optimizers.Nadam(lr=4e-5, beta_1=0.9, beta_2=0.999),
                  loss=['binary_crossentropy', CondBCE(), CondBCE(), CondBCE(), CondBCE(), CondBCE()],
                  metrics=[accuracy, auc_roc],
                  loss_weights=[0.4, 1, 1, 1, 1, 1])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_size,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=val_size,
                        initial_epoch=initial_epoch,
                        callbacks=[tensorboard_callback, checkpoint, ])


def main(batch_size=8, img_dim=(331, 331), epochs=40, load_saved_model=False):
    # pre-instantiations
    train_gen, val_gen, train_size, val_size = data_generators(batch_size, img_dim)
    if load_saved_model:
        print("Loading model savepoint")
        model = load_model(MODEL_CHECKPOINT, custom_objects={'auc_roc': auc_roc})
    else:
        # ...weil synthetische regularisierungsmaßnahme
        # model = classifier(create_model(img_dim))
        model = create_model(img_dim)

    # multi GPU case :))
    # model = multi_gpu_model(model)

    # training
    train(model, epochs, train_gen, val_gen, train_size, val_size)


if __name__ == '__main__':
    main()
