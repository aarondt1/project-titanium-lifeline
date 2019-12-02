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
    img_data_gen = keras.preprocessing.image.ImageDataGenerator(  # rotation_range=7,
        width_shift_range=0.04,
        height_shift_range=0.04,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1 / 255)
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
    return train_gen, val_gen


def create_model(img_dim):
    backbone = keras.applications.nasnet.NASNetLarge(input_shape=(*img_dim, 1), include_top=False, weights=None,
                                                      pooling=None)
    weights_path = keras.utils.get_file(
        'nasnet_mobile_no_top.h5',
        keras_applications.nasnet.NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='1ed92395b5b598bdda52abe5c0dbfd63')
    backbone.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return backbone


def classifier(model):
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    x = keras.layers.Dense(14, activation='sigmoid')(x)
    clsfr = keras.Model(model.input, x)
    return clsfr


def train(model, epochs, train_gen, val_gen):
    filepath = './output/model_durchlauf1_LARGE.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    logdir = "./logs/scalars/"  # /scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_gen),
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_gen),
                        initial_epoch=0,
                        callbacks=[tensorboard_callback, checkpoint])


def main(tpu_training=False, batch_size=16, img_dim=(331, 331), epochs=40):
    # pre-instantiations
    train_gen, val_gen = data_generators(batch_size, img_dim)
    model = classifier(create_model(img_dim))
    model.compile(keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='binary_crossentropy',
                  metrics=['acc', auc_roc])

    # ONLY REQUIRED for training with TPU
    if tpu_training:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

    # training
    train(model, epochs, train_gen, val_gen)


if __name__ == '__main__':
    main()

