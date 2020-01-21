import tensorflow as tf

# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    y_true = tf.maximum(y_true, 0.0)
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def acc(y_true, y_pred):
    y_true = tf.maximum(y_true, 0.0)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.cast(tf.equal(y_true, tf.cast(y_pred >= 0.5, tf.float32)), tf.keras.backend.floatx())
