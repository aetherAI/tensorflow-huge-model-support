# Tensorflow Huge Model Support (HMS)

This library is designed to speed up huge model training on unified memory. 
It takes a computation graph built by the user, conducts analysis, implements group execution and prefetch by editing the graph.
A callback hook is provided to easily apply HMS on a tf.keras model.

## Requirements

- Tensorflow v1 (tensorflow-gpu==1.15.3)
- GCC >= 7

## Installation

To install HMS, simply run the following commands:
```
[CUDA_PATH=YOU_CUDA_PATH] pip install .
```
, where CUDA_PATH is /usr/local/cuda by default.

## Usage

HMS can be simply applied on tf.keras model by a callback function, as described below.

1. Import HMS tf_keras module.
```python
from tensorflow_huge_model_support.tf_keras import init, HMSTFKerasCallback
```

2. Call init before model building(, and after horovod initializes).

Without horovod:
```python
init()
```

With horovod:
```python
import horovod.tensorflow.keras as hvd
hvd.init()
init(hvd=hvd)
```

3.  Define a HMSKerasCallback.
```python
hms_callback = HMSTFKerasCallback(
    hvd=hvd,
    default_batch_size=DEFAULT_BATCH_SIZE
)
```
, where `hvd` can be skipped if not using Horovod. 

4. Pass the callback to the Keras fit or fit_generator function.
```python
model.fit_generator(..., callbacks=[hms_callback] + OTHER_CALLBACKS, ...)
```

Note: Don't forget to add hvd.callbacks.BroadcastGlobalVariablesCallback(0) in the callback list if using Horovod.
