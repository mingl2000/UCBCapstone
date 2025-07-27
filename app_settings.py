import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

def set_random_seed():
    """
    Set random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()


def set_global_settings():
    """
    Set global settings for the application.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    warnings.filterwarnings('ignore', message='.*use_unbounded_threadpool.*')  # Suppress specific warnings
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
