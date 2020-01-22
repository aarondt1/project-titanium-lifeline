import warnings
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf

#tf.get_logger().setLevel('INFO')
#import logging

#tf.get_logger().setLevel(logging.ERROR)
import keras
import keras_applications
from metrics import auc_roc, acc
from keras import backend as K
from keras.models import load_model
from keras_contrib.applications.nasnet import NASNetLarge, NASNetMobile, NASNET_LARGE_WEIGHT_PATH_WITH_auxiliary, NASNET_MOBILE_WEIGHT_PATH_WITH_AUXULARY
from keras.utils import multi_gpu_model
import itertools
from keras.applications.mobilenet_v2 import MobileNetV2

MODEL_CHECKPOINT = "./output/model_durchlauf2_LARGE.15-0.44.h5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
K.tensorflow_backend.set_session(tf.Session(config=config))

df_train = pd.read_csv('CheXpert-v1.0-small/train.csv')#.fillna(0)
#  converting -1 to random interval
# ran = pd.DataFrame(np.random.uniform(0.55, 0.85, size=(len(df_train.index), len(df_train.columns))),
#                    columns=df_train.columns, index=df_train.index)
# df_train = df_train.where(df_train != -1.0, ran)

# no post-processing necessary due to the intended lack of uncertainty labels / nans
df_val = pd.read_csv('CheXpert-v1.0-small/valid.csv')


def generate_all_possible_combinations(df, n=14):  # reduced valid combs from 16,384 to 2,402
    comb = np.array(list(itertools.product([0, 1], repeat=n)))
    comb = comb[2:]  # all zeros (excl. SupportDevices) are invalid as well

    # exclusion: no_finding
    for i in range(1, 13):
        comb = comb[~((comb[:, 0] == 1) & (comb[:, i] == 1))]

    # hierachy: lung opactity
    for i in range(4, 9):
        comb = comb[~((comb[:, 3] == 0) & (comb[:, i] == 1))]
    df = df[~((df['Atelectasis'] == 1) & (df['Lung Opacity'] == 0))]  # 349
    df.loc[(df['Atelectasis'] == 1) & (df['Lung Opacity'] == -1), 'Lung Opacity'] = 1
    df.loc[(df['Atelectasis'] == 1) & (df['Lung Opacity'].isna()), 'Lung Opacity'] = 1
    df = df[~((df['Edema'] == 1) & (df['Lung Opacity'] == 0))]  # 448
    df.loc[(df['Edema'] == 1) & (df['Lung Opacity'] == -1), 'Lung Opacity'] = 1
    df.loc[(df['Edema'] == 1) & (df['Lung Opacity'].isna()), 'Lung Opacity'] = 1
    df = df[~((df['Lung Lesion'] == 1) & (df['Lung Opacity'] == 0))]  # 138
    df.loc[(df['Lung Lesion'] == 1) & (df['Lung Opacity'] == -1), 'Lung Opacity'] = 1
    df.loc[(df['Lung Lesion'] == 1) & (df['Lung Opacity'].isna()), 'Lung Opacity'] = 1
    df = df[~((df['Pneumonia'] == 1) & (df['Lung Opacity'] == 0))]  # 129
    df.loc[(df['Pneumonia'] == 1) & (df['Lung Opacity'] == -1), 'Lung Opacity'] = 1
    df.loc[(df['Pneumonia'] == 1) & (df['Lung Opacity'].isna()), 'Lung Opacity'] = 1
    df = df[~((df['Consolidation'] == 1) & (df['Lung Opacity'] == 0))]  # 61
    df.loc[(df['Consolidation'] == 1) & (df['Lung Opacity'] == -1), 'Lung Opacity'] = 1
    df.loc[(df['Consolidation'] == 1) & (df['Lung Opacity'].isna()), 'Lung Opacity'] = 1

    # hierachy: consolidation
    comb = comb[~((comb[:, 6] == 0) & (comb[:, 7] == 1))]
    df = df[~((df['Pneumonia'] == 1) & (df['Consolidation'] == 0))]  # 528
    df.loc[(df['Pneumonia'] == 1) & (df['Consolidation'] == -1), 'Consolidation'] = 1
    df.loc[(df['Pneumonia'] == 1) & (df['Consolidation'].isna()), 'Consolidation'] = 1

    # hierachy: cardiomegaly
    comb = comb[~((comb[:, 1] == 0) & (comb[:, 2] == 1))]
    df = df[~((df['Cardiomegaly'] == 1) & (df['Enlarged Cardiomediastinum'] == 0))]  # 172
    df.loc[(df['Cardiomegaly'] == 1) & (df['Enlarged Cardiomediastinum'] == -1), 'Enlarged Cardiomediastinum'] = 1
    df.loc[(df['Cardiomegaly'] == 1) & (df['Enlarged Cardiomediastinum'].isna()), 'Enlarged Cardiomediastinum'] = 1

    return comb, df


combs, df_train = generate_all_possible_combinations(df_train)
df_train = df_train.fillna(-1)
df_train = df_train.loc[(df_train.iloc[:, 5:] != -1).any(axis=1)]  # 2560 rows
# df_train = df_train[:10000]


class HEXLoss(keras.losses.Loss):
    def __init__(self):
        super(HEXLoss, self).__init__()
        self.comb = tf.convert_to_tensor(combs, tf.float32)

    def call(self, y_true, y_pred):  # (bs, classes)
        y_true = K.reshape(K.cast(y_true, y_pred.dtype), (-1, y_pred.shape[1]))

        def for_each_batch(args):
            y_true, y_pred = args

            y_pred_ = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
            y_true_ = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
            comb_ = tf.gather(self.comb, tf.where(tf.not_equal(y_true, -1))[:,0], axis=1)
            y_true_combs = tf.gather(self.comb, tf.where(tf.equal(comb_, y_true_))[:,0])
            # yp = K.sum(K.log(y_true_*y_pred_ + (1-y_true_)*(1-y_pred_)))
            yp = K.logsumexp(K.sum(-K.binary_crossentropy(y_true_combs, y_pred_), axis=1))

            # certain_combs = tf.numpy_function(lambda x: np.unique(x, axis=0), [tf.boolean_mask(self.comb, tf.not_equal(y_true, -1), axis=1)], tf.float32)
            # certain_combs = tf.Print(certain_combs, [certain_combs], 'Combs ')
            # yp -= K.logsumexp(K.sum(K.log(y_pred*self.comb + (1-y_pred)*(1-self.comb)), axis=1))
            yp -= K.logsumexp(K.sum(-K.binary_crossentropy(self.comb, y_pred_), axis=1))
            return yp

        yp = tf.map_fn(for_each_batch, (y_true, y_pred), dtype=tf.float32)
        # yp = tf.Print(yp, [yp], 'yp after: ')

        return -yp


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
            yield iter_gen[0], iter_gen[1]  # [iter_gen[1], iter_gen[1]]

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
    backbone = MobileNetV2(input_shape=(*img_dim, 1),
                           # dropout=0.5,
                           # weight_decay=5e-5,
                           # use_auxiliary_branch=True,
                           include_top=False,
                           weights=None,
                           input_tensor=None,
                           pooling='avg',
                           classes=14,
                           alpha=1.4,
                           # activation='sigmoid'
                           )
    x = keras.layers.Dropout(0.5, name='dropout')(backbone.output)
    x = keras.layers.Dense(14, activation='sigmoid', use_bias=True, name='Logits')(x)

    BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                        'releases/download/v1.1/')
    MODEL_NAME = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5'
    weights_path = keras.utils.get_file(
        MODEL_NAME,
        BASE_WEIGHT_PATH + MODEL_NAME,
        cache_subdir='models')
    backbone.load_weights(weights_path, by_name=True, skip_mismatch=True)

    backbone = keras.models.Model(inputs=backbone.inputs, outputs=x)

    return backbone


# set initial_epoch to last successful epoch
def train(model, epochs, train_gen, val_gen, train_size, val_size, initial_epoch=0):
    filepath = './output/model_hex_MOBILENET.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                 mode='min')
    logdir = "./logs/scalars/hex/"  # /scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tensorboard_callback.samples_seen = initial_epoch  # * len(train_gen)
    tensorboard_callback.samples_seen_at_last_write = tensorboard_callback.samples_seen

    model.compile(keras.optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999),
                  loss=HEXLoss(),
                  metrics=[acc, auc_roc])
                  #loss_weights=[1, 0.4])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_size,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=val_size,
                        initial_epoch=initial_epoch,
                        callbacks=[tensorboard_callback, checkpoint, ])


def main(batch_size=64, img_dim=(224, 224), epochs=40, load_saved_model=False):
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
