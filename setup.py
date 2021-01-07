from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
import subprocess
import tensorflow as tf
import os

from Cython.Build import cythonize # cythonize must be imported after setuptools

if "CUDA_PATH" in os.environ:
    CUDA_PATH = os.environ["CUDA_PATH"]
else:
    CUDA_PATH = '/usr/local/cuda'

install_requires = [
    'gputil',
    'networkx==2.2',
    'psutil',
    'pycrypto',
    'Cython'
]

setup(
    name='tensorflow-huge-model-support',
    version='1.1',
    description='',
    author='Chi-Chung Chen',
    author_email='chenchc@aetherai.com',
    ext_modules=(
        cythonize('tensorflow_huge_model_support/*.py', compiler_directives={'language_level' : "3"}) + 
        [Extension(
            'tensorflow_huge_model_support.prefetch', 
            ['prefetch_module/prefetch.cc'],
            include_dirs=[CUDA_PATH + '/include/'],
            library_dirs=[CUDA_PATH + '/lib64/'],
            libraries=['cuda'],
            extra_compile_args=['-fPIC', '-std=c++11', '-O3'] + tf.sysconfig.get_compile_flags(),
            extra_link_args=tf.sysconfig.get_link_flags(),
            language='c++',
        )]
    ),
    install_requires=install_requires,
)

