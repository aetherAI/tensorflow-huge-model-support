import os
from tensorflow_huge_model_support.utils import USE_TF_2, get_gpu_mem_size, get_host_mem_size, create_session
from tensorflow_huge_model_support.hms import HMS

def HMSSession(config=None, hvd=None, hooks=None):
    host_mem_size = get_host_mem_size()
    gpu_mem_size = get_gpu_mem_size(hvd=hvd)
    um_ratio = host_mem_size / gpu_mem_size
    sess = create_session(
        base_config=config,
        um_ratio=um_ratio,
        visible_devices=(str(hvd.local_rank()) if hvd != None else None),
        hooks=hooks
    )

    # Increase the limit of max workspace size of cuDNN covolutional layers.
    # This setting prevents cuDNN from using slow but memory-conservative algorithms.
    if 'TF_CUDNN_WORKSPACE_LIMIT_IN_MB' in os.environ:
        print('[HMS] Overriding TF_CUDNN_WORKSPACE_LIMIT_IN_MB...')
    os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = str(host_mem_size)

    return sess
