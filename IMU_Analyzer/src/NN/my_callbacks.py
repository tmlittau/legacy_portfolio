from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import io
import json
import os
import time

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
from tensorflow.python.util.tf_export import tf_export


try:
    import requests
except ImportError:
    requests = None


def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode='train'):

    # Check if callbacks have already been configured.
    if isinstance(callbacks, CallbackList):
        return callbacks

    if not callbacks:
        callbacks = []

    # Add additional callbacks during training.
    if mode == 'train':
        model.history = History()
        stateful_metric_names = None
        if hasattr(model, 'metrics_names'):
            stateful_metric_names = model.metrics_names[1:]  # Exclude `loss`
        callbacks = [BaseLogger(stateful_metrics=stateful_metric_names)
        ] + (callbacks or []) + [model.history]
        if verbose:
            callbacks.append(
                ProgbarLogger(count_mode, stateful_metrics=stateful_metric_names))
    callback_list = CallbackList(callbacks)

    # Set callback model
    callback_model = model._get_callback_model()
    callback_list.set_model(callback_model)

    # Set callback parameters
    callback_metrics = []
    # When we have deferred build scenario with iterator input, we will compile
    # when we standardize first batch of data.
    if mode != 'predict' and hasattr(model, 'metrics_names'):
        callback_metrics = copy.copy(model.metrics_names)
        if do_validation:
            callback_metrics += ['val_' + n for n in model.metrics_names]
    callback_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    }
    callback_list.set_params(callback_params)

    if (do_validation and not model._distribution_strategy and
        not model.run_eagerly):
        # Need to create the eval_function before start of the first epoch
        # because TensorBoard callback on_epoch_begin adds summary to the
        # list of fetches of the eval_function
        callback_model._make_eval_function()

    callback_list.model.stop_training = False
    return callback_list

def _is_generator_like(data):
    """Checks if data is a generator, Sequence, or Iterator."""
    return (hasattr(data, 'next') or hasattr(data, '__next__') or isinstance(
        data, (Sequence, iterator_ops.Iterator, iterator_ops.EagerIterator)))


class CallbackList(object):


    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.params = {}
        self.model = None
        self._reset_batch_timing()

    def _reset_batch_timing(self):
        self._delta_t_batch = 0.
        self._delta_ts = collections.defaultdict(
            lambda: collections.deque([], maxlen=self.queue_length))

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        # TODO(omalleyt): add batch hooks for test/predict.
        if mode != 'train':
            return

        hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
        if hook == 'begin':
            self._t_enter_batch = time.time()
        if hook == 'end':
            # Batch is ending, calculate batch time.
            self._delta_t_batch = time.time() - self._t_enter_batch

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)
        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

        delta_t_median = np.median(self._delta_ts[hook_name])
        if (self._delta_t_batch > 0. and
            delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            logging.warning(
                'Method (%s) is slow compared '
                'to the batch update (%f). Check your callbacks.', hook_name,
                delta_t_median)

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        # TODO(omalleyt): add test/predict methods.
        if mode == 'train':
            self.on_train_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        # TODO(omalleyt): add test/predict methods.
        if mode == 'train':
            self.on_train_end()

    def on_batch_begin(self, batch, logs=None):
        self._call_batch_hook('train', 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self._call_batch_hook('train', 'end', batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None, mode='train'):
        """Called at the start of an epoch.
        Arguments:
          epoch: integer, index of epoch.
          logs: dictionary of logs.
           mode: One of 'train'/'test'/'predict'
        """
        if mode == 'train':
            logs = logs or {}
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs)
        self._reset_batch_timing()

    def on_epoch_end(self, epoch, logs=None, mode='train'):
        """Called at the end of an epoch.
        Arguments:
           epoch: integer, index of epoch.
           logs: dictionary of logs.
           mode: One of 'train'/'test'/'predict'
        """
        if mode == 'train':
            logs = logs or {}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Arguments:
           batch: integer, index of batch within the current epoch.
           logs: dictionary of logs.
        """
        self._call_batch_hook('train', 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Arguments:
           batch: integer, index of batch within the current epoch.
           logs: dictionary of logs.
        """
        self._call_batch_hook('train', 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Arguments:
           logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Arguments:
           logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


@tf_export('keras.callbacks.Callback')
class Callback(object):

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        # For backwards compatibility
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        # For backwards compatibility
        self.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
    

class Histories(Callback):
  
    def __init__(self,
                 mainWindow,
                 filepath,
                 count_mode='samples',
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1,
                 stateful_metrics=None):
          
        super(Histories, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown "count_mode": ' + str(count_mode))
        self.stateful_metrics = set(stateful_metrics or [])
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.mainWindow = mainWindow

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, epoch, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        if self.use_steps:
            self.target = self.params['steps']
        else:
            self.target = self.params['samples']

        if self.verbose:
            if self.epochs > 1:
                print('Epoch %d/%d' % (epoch + 1, self.epochs))
                self.mainWindow.write_to_terminal('[Model] Epoch %d/%d' % (epoch + 1, self.epochs))

        #self.mainWindow.init_progbar()
        
    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        self.mainWindow.update_progbar(self.seen)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                            self.mainWindow.write_to_terminal('[Model] Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
                            self.mainWindow.write_to_terminal('[Model] Epoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    self.mainWindow.write_to_terminal('[Model] Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
