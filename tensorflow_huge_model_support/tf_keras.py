import os
from tensorflow_huge_model_support.hms import HMS
from tensorflow_huge_model_support.utils import USE_TF_2, get_gpu_mem_size, get_host_mem_size, create_session

import tensorflow as tf

def init(base_config=None, hvd=None, copy=False):
    # Let Tensorflow use Unified Memory.
    host_mem_size = get_host_mem_size()
    gpu_mem_size = get_gpu_mem_size(hvd=hvd)
    um_ratio = host_mem_size / gpu_mem_size
    sess = create_session(
        base_config=base_config,
        um_ratio=um_ratio, 
        visible_devices=(str(hvd.local_rank()) if hvd != None else None)
    )

    if copy:
        old_sess = tf.keras.backend.get_session()
        var_list = tf.global_variables()
        if var_list != []:
            var_is_inited_list = [tf.is_variable_initialized(var) for var in var_list]
            var_is_inited_val_list = old_sess.run(var_is_inited_list)
            var_inited_var_list = [var for i, var in enumerate(var_list) if var_is_inited_val_list[i]]
            if var_inited_var_list != []:
                var_inited_var_val_list = old_sess.run(var_inited_var_list)
                sess.run([tf.assign(var_inited_var_list[i], var_inited_var_val_list[i]) for i in range(len(var_inited_var_list))])

    tf.keras.backend.set_session(sess)

    return sess

class HMSTFKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, hvd=None, **kwargs):
        self._hvd = hvd
        self._kwargs = kwargs
        self._hms = None

    def set_model(self, model):
        if self._hms != None:
            return
        #train_function = model._fit_function if hasattr(model, '_fit_function') else model.train_function
        model._make_train_function()
        train_function = model.train_function
        assert train_function != None

        # Invoke HMS
        fwd_op = model.total_loss.op
        bwd_op = train_function.updates_op
        self._hms = hms = HMS(fwd_op, bwd_op, hvd=self._hvd, **self._kwargs)
        hms.run()

        # Inialize HMS variables
        old_sess = tf.keras.backend.get_session()
        old_config = old_sess._config

        sess = init(base_config=old_config, hvd=self._hvd, copy=True)
        hms_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='HMS/')
        sess.run(tf.initializers.variables(hms_vars))
