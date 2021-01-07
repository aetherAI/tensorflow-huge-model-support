import copy
import GPUtil
import os
import psutil
import tensorflow as tf

USE_TF_2 = tf.__version__.startswith('2')

def get_gpu_mem_size(gpu_id=0, hvd=None):
    if hvd:
        gpu_id = hvd.local_rank()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpu_list = [int(gpu) for gpu in gpu_list]
        gpu_mem_size = GPUtil.getGPUs()[gpu_list[gpu_id]].memoryTotal
    else:
        gpu_mem_size = GPUtil.getGPUs()[gpu_id].memoryTotal

    gpu_mem_size *= 1024 * 1024

    return gpu_mem_size
    
def get_host_mem_size():
    return psutil.virtual_memory().total

def create_session(base_config=None, um_ratio=None, visible_devices=None, hooks=None):
    if USE_TF_2:
        ConfigProto = tf.compat.v1.ConfigProto
        Session = tf.compat.v1.Session
        MonitoredTrainingSession = tf.compat.v1.train.MonitoredTrainingSession
    else:
        ConfigProto = tf.ConfigProto
        Session = tf.Session
        MonitoredTrainingSession = tf.train.MonitoredTrainingSession
    
    if base_config:
        config = copy.deepcopy(base_config)
    else:
        config = ConfigProto()

    if um_ratio:
        config.gpu_options.per_process_gpu_memory_fraction = um_ratio
    if visible_devices:
        config.gpu_options.visible_device_list = visible_devices

    config.gpu_options.allow_growth = True

    if hooks:
        sess = MonitoredTrainingSession(config=config, hooks=hooks)
    else:
        sess = Session(config=config)

    return sess
