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
import matplotlib.pyplot as plt
from datetime import datetime
from metrics import auc_roc


df_train = pd.read_csv('CheXpert-v1.0-small/train.csv').fillna(0)  # [:500]
# converting -1 to random interval
ran = pd.DataFrame(np.random.uniform(0.55, 0.85, size=(len(df_train.index), len(df_train.columns))),
                   columns=df_train.columns, index=df_train.index)
df_train = df_train.where(df_train != -1.0, ran)

df_val = pd.read_csv('CheXpert-v1.0-small/valid.csv').fillna(0)  # [:500]
# converting -1 to random interval
ran = pd.DataFrame(np.random.uniform(0.55, 0.85, size=(len(df_val.index), len(df_val.columns))), columns=df_val.columns,
                   index=df_val.index)
df_val = df_val.where(df_val != -1.0, ran)


def data_generators():
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
                                                 target_size=(224, 224),
                                                 color_mode='grayscale',
                                                 class_mode='raw',
                                                 batch_size=64,
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
                                               target_size=(224, 224),
                                               color_mode='grayscale',
                                               class_mode='raw',
                                               batch_size=64,
                                               shuffle=True,
                                               interpolation='box')
    return train_gen, val_gen


def create_model():
    backbone = keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 1), include_top=False, weights=None,
                                                      pooling=None)
    weights_path = keras.utils.get_file(
        'nasnet_mobile_no_top.h5',
        keras_applications.nasnet.NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='1ed92395b5b598bdda52abe5c0dbfd63')
    backbone.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return backbone


def classifier(model):
    b1 = keras.layers.GlobalAveragePooling2D()(model.output)
    b1 = keras.layers.Dense(14, activation='sigmoid')(b1)
    b1 = keras.Model(model.input, b1)
    b1.compile(keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    return b1


def train(model):
    filepath = "./output/model_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    logdir = "./logs/scalars/"  # /scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_gen),
                        epochs=10,
                        # steps_per_epoch=15,
                        validation_data=val_gen,
                        validation_steps=len(val_gen),
                        initial_epoch=0,
                        callbacks=[tensorboard_callback, checkpoint])


def main(tpu_training=False):
  train_gen, val_gen = data_generators()
  backbone_model = create_model()
  model = classifier(backbone_model)

  # ONLY REQUIRED for training with TPU
  if tpu_training:
    model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
  
  train(model)


if __name__ == '__main__':
    main()

