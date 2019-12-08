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
from keras_contrib.applications.nasnet import NASNetLarge

MODEL_CHECKPOINT = "./output/model_durchlauf2_LARGE.10-1.33.h5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
K.tensorflow_backend.set_session(tf.Session(config=config))

df_train = pd.read_csv('CheXpert-v1.0-small/train.csv').fillna(0)  # [:500]
# converting -1 to random interval
ran = pd.DataFrame(np.random.uniform(0.55, 0.85, size=(len(df_train.index), len(df_train.columns))),
                   columns=df_train.columns, index=df_train.index)
df_train = df_train.where(df_train != -1.0, ran)

# no post-processing necessary due to the intended lack of uncertainty labels / nans
df_val = pd.read_csv('CheXpert-v1.0-small/valid.csv')  # [:500]


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
        for iter_gen in gen:
            yield iter_gen[0], [iter_gen[1], iter_gen[1]]

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

    # weights_path = keras.utils.get_file(
    #     'nasnet_large_no_top.h5',
    #     keras_applications.nasnet.NASNET_LARGE_WEIGHT_PATH_NO_TOP,
    #     cache_subdir='models',
    #     file_hash='d81d89dc07e6e56530c4e77faddd61b5')
    backbone.load_weights(MODEL_CHECKPOINT, by_name=True, skip_mismatch=True)
    return backbone


def classifier(model):
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    x = keras.layers.Dense(14, activation='sigmoid')(x)
    clsfr = keras.Model(model.input, x)
    return clsfr


# set initial_epoch to last successful epoch
def train(model, epochs, train_gen, val_gen, train_size, val_size, initial_epoch=10):
    filepath = './output/model_durchlauf2_LARGE.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    logdir = "./logs/scalars/"  # /scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tensorboard_callback.samples_seen = initial_epoch  # * len(train_gen)
    tensorboard_callback.samples_seen_at_last_write = tensorboard_callback.samples_seen

    model.compile(keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999), loss='binary_crossentropy',
                  metrics=['acc', auc_roc],
                  loss_weights=[1, 0.4])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_size,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=val_size,
                        initial_epoch=initial_epoch,
                        callbacks=[tensorboard_callback, checkpoint, ])


def main(tpu_training=False, batch_size=8, img_dim=(331, 331), epochs=40, load_saved_model=True):
    # pre-instantiations
    train_gen, val_gen, train_size, val_size = data_generators(batch_size, img_dim)
    if load_saved_model:
        print("Loading model savepoint")
        model = load_model(MODEL_CHECKPOINT, custom_objects={'auc_roc': auc_roc})
    else:
        # ...weil synthetische regularisierungsmaßnahme
        # model = classifier(create_model(img_dim))
        model = create_model(img_dim)

    # ONLY REQUIRED for training with TPU
    if tpu_training:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    # training
    train(model, epochs, train_gen, val_gen, train_size, val_size)


if __name__ == '__main__':
    main()
