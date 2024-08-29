"""
CIKM'2021：One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction

https://arxiv.org/abs/2101.11427
"""
from functools import partial
from typing import List, Callable, Optional, Dict, Type

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

from ..utils.core import dnn_layer, dice
from ..utils.interaction import AttentionBase
from ..utils.type_declaration import Field


class PartitionedNormalization(tf.layers.BatchNormalization):

    def __init__(self,
                 num_domain,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=init_ops.zeros_initializer(),
                 gamma_initializer=init_ops.ones_initializer(),
                 moving_mean_initializer=init_ops.zeros_initializer(),
                 moving_variance_initializer=init_ops.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 trainable=True,
                 virtual_batch_size=None,
                 adjustment=None,
                 name=None,
                 **kwargs):
        super(PartitionedNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            fused=fused,
            trainable=trainable,
            virtual_batch_size=virtual_batch_size,
            adjustment=adjustment,
            name=name,
            **kwargs)

        self.num_domain = num_domain

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        if self.virtual_batch_size is not None:
            if self.virtual_batch_size <= 0:
                raise ValueError('virtual_batch_size must be a positive integer that '
                                 'divides the true batch size of the input Tensor')
            # If using virtual batches, the first dimension must be the batch
            # dimension and cannot be the batch norm axis
            if 0 in self.axis:
                raise ValueError('When using virtual_batch_size, the batch dimension '
                                 'must be 0 and thus axis cannot include 0')
            if self.adjustment is not None:
                raise ValueError('When using virtual_batch_size, adjustment cannot '
                                 'be specified')

        if self.fused in (None, True):
            # TODO(yaozhang): if input is not 4D, reshape it to 4D and reshape the
            # output back to its original shape accordingly.
            if self._USE_V2_BEHAVIOR:
                if self.fused is None:
                    self.fused = (ndims == 4)
                elif self.fused and ndims != 4:
                    raise ValueError('Batch normalization layers with fused=True only '
                                     'support 4D input tensors.')
            else:
                assert self.fused is not None
                self.fused = (ndims == 4 and self._fused_can_be_used())
            # TODO(chrisying): fused batch norm is currently not supported for
            # multi-axis batch norm and by extension virtual batches. In some cases,
            # it might be possible to use fused batch norm but would require reshaping
            # the Tensor to 4D with the axis in 1 or 3 (preferred 1) which is
            # particularly tricky. A compromise might be to just support the most
            # common use case (turning 5D w/ virtual batch to NCHW)

        if self.fused:
            if self.axis == [1]:
                self._data_format = 'NCHW'
            elif self.axis == [3]:
                self._data_format = 'NHWC'
            else:
                raise ValueError('Unsupported axis, fused batch norm only supports '
                                 'axis == [1] or axis == [3]')

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                 input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]
            if self.virtual_batch_size is not None:
                # When using virtual batches, add an extra dim at index 1
                param_shape.insert(1, 1)
                for idx, x in enumerate(self.axis):
                    self.axis[idx] = x + 1  # Account for added dimension

        param_shape = list(param_shape)
        origin_param_shape = param_shape.copy()
        param_shape.insert(0, self.num_domain)

        if self.scale:
            self.gamma = self.add_weight(
                name='domain_gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.global_gamma = self.add_weight(
                name='global_gamma',
                shape=origin_param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            if self.fused:
                self._gamma_const = K.constant(
                    1.0, dtype=self._param_dtype, shape=param_shape)

        if self.center:
            self.beta = self.add_weight(
                name='domain_beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
            self.global_beta = self.add_weight(
                name='global_beta',
                shape=origin_param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None
            if self.fused:
                self._beta_const = K.constant(
                    0.0, dtype=self._param_dtype, shape=param_shape)

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)

            self.moving_variance = self.add_weight(
                name='moving_variance',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)

            if self.renorm:
                # In batch renormalization we track the inference moving stddev instead
                # of the moving variance to more closely align with the paper.
                def moving_stddev_initializer(*args, **kwargs):
                    return math_ops.sqrt(
                        self.moving_variance_initializer(*args, **kwargs))

                with distribution_strategy_context.get_strategy(
                ).extended.colocate_vars_with(self.moving_variance):
                    self.moving_stddev = self.add_weight(
                        name='moving_stddev',
                        shape=param_shape,
                        dtype=self._param_dtype,
                        initializer=moving_stddev_initializer,
                        synchronization=tf_variables.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf_variables.VariableAggregation.MEAN,
                        experimental_autocast=False)

                # Create variables to maintain the moving mean and standard deviation.
                # These are used in training and thus are different from the moving
                # averages above. The renorm variables are colocated with moving_mean
                # and moving_stddev.
                # NOTE: below, the outer `with device` block causes the current device
                # stack to be cleared. The nested ones use a `lambda` to set the desired
                # device and ignore any devices that may be set by the custom getter.
                def _renorm_variable(name,
                                     shape,
                                     initializer=init_ops.zeros_initializer()):
                    """Create a renorm variable."""
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        dtype=self._param_dtype,
                        initializer=initializer,
                        synchronization=tf_variables.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf_variables.VariableAggregation.MEAN,
                        experimental_autocast=False)
                    return var

                with distribution_strategy_context.get_strategy(
                ).extended.colocate_vars_with(self.moving_mean):
                    self.renorm_mean = _renorm_variable('renorm_mean', param_shape,
                                                        self.moving_mean_initializer)
                with distribution_strategy_context.get_strategy(
                ).extended.colocate_vars_with(self.moving_stddev):
                    self.renorm_stddev = _renorm_variable('renorm_stddev', param_shape,
                                                          moving_stddev_initializer)
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _fused_batch_norm(self, inputs, domain_index, training):
        """Returns the output of fused batch norm."""
        beta = self.beta[domain_index] + self.global_beta if self.center else self._beta_const
        gamma = self.gamma[domain_index] * self.global_gamma if self.scale else self._gamma_const

        domain_moving_mean = self.moving_mean[domain_index]
        domain_moving_variance = self.moving_variance[domain_index]
        domain_moving_stddev = self.moving_stddev[domain_index]

        # TODO(b/129279393): Support zero batch input in non DistributionStrategy
        # code as well.
        if self._support_zero_size_input():
            inputs_size = array_ops.size(inputs)
        else:
            inputs_size = None

        def _fused_batch_norm_training():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                epsilon=self.epsilon,
                data_format=self._data_format)

        def _fused_batch_norm_inference():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=domain_moving_mean,
                variance=domain_moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format)

        output, mean, variance = tf_utils.smart_cond(
            training, _fused_batch_norm_training, _fused_batch_norm_inference)
        if not self._bessels_correction_test_only:
            # Remove Bessel's correction to be consistent with non-fused batch norm.
            # Note that the variance computed by fused batch norm is
            # with Bessel's correction.
            sample_size = math_ops.cast(
                array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
            factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
            variance *= factor

        training_value = tf_utils.constant_value(training)
        if training_value is None:
            momentum = tf_utils.smart_cond(training,
                                           lambda: self.momentum,
                                           lambda: 1.0)
        else:
            momentum = ops.convert_to_tensor(self.momentum)
        if training_value or training_value is None:
            def mean_update():
                return self._assign_moving_average(domain_moving_mean, mean, momentum,
                                                   inputs_size)

            def variance_update():
                """Update self.moving_variance with the most recent data point."""
                if self.renorm:
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = self._assign_moving_average(
                        domain_moving_stddev, math_ops.sqrt(variance + self.epsilon),
                        momentum, inputs_size)
                    return self._assign_new_value(
                        domain_moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon))
                else:
                    return self._assign_moving_average(domain_moving_variance, variance,
                                                       momentum, inputs_size)

            self.add_update(mean_update)
            self.add_update(variance_update)

        return output

    def _renorm_correction_and_moments(self, domain_index, mean, variance, training,
                                       inputs_size):
        """Returns the correction and update values for renorm."""
        stddev = math_ops.sqrt(variance + self.epsilon)
        # Compute the average mean and standard deviation, as if they were
        # initialized with this batch's moments.
        domain_renorm_mean = self.renorm_mean[domain_index]
        # Avoid divide by zero early on in training.
        domain_renorm_stddev = math_ops.maximum(self.renorm_stddev[domain_index],
                                         math_ops.sqrt(self.epsilon))
        # Compute the corrections for batch renorm.
        r = stddev / domain_renorm_stddev
        d = (mean - domain_renorm_mean) / domain_renorm_stddev
        # Ensure the corrections use pre-update moving averages.
        with ops.control_dependencies([r, d]):
            mean = array_ops.identity(mean)
            stddev = array_ops.identity(stddev)
        rmin, rmax, dmax = [self.renorm_clipping.get(key)
                            for key in ['rmin', 'rmax', 'dmax']]
        if rmin is not None:
            r = math_ops.maximum(r, rmin)
        if rmax is not None:
            r = math_ops.minimum(r, rmax)
        if dmax is not None:
            d = math_ops.maximum(d, -dmax)
            d = math_ops.minimum(d, dmax)
        # When not training, use r=1, d=0.
        r = tf_utils.smart_cond(training, lambda: r, lambda: array_ops.ones_like(r))
        d = tf_utils.smart_cond(training,
                                lambda: d,
                                lambda: array_ops.zeros_like(d))

        def _update_renorm_variable(var, value, inputs_size):
            """Updates a moving average and weight, returns the unbiased value."""
            value = array_ops.identity(value)

            def _do_update():
                """Updates the var, returns the updated value."""
                new_var = self._assign_moving_average(var, value, self.renorm_momentum,
                                                      inputs_size)
                return new_var

            def _fake_update():
                return array_ops.identity(var)

            return tf_utils.smart_cond(training, _do_update, _fake_update)

        # TODO(yuefengz): colocate the operations
        update_new_mean = _update_renorm_variable(domain_renorm_mean, mean,
                                                  inputs_size)
        update_new_stddev = _update_renorm_variable(domain_renorm_stddev, stddev,
                                                    inputs_size)

        # Update the inference mode moving averages with the batch value.
        with ops.control_dependencies([update_new_mean, update_new_stddev]):
            out_mean = array_ops.identity(mean)
            out_variance = array_ops.identity(variance)

        return (r, d, out_mean, out_variance)

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (
                                       variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign(variable, variable - update_delta, name=scope)

    def call(self, inputs, domain_index, training=None):
        training = self._get_training_value(training)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = array_ops.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = array_ops.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, domain_index, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale = _broadcast(self.gamma[domain_index] * self.global_gamma)
        offset = _broadcast(self.beta[domain_index] + self.global_beta)
        domain_moving_mean = self.moving_mean[domain_index]
        domain_moving_variance = self.moving_variance[domain_index]

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = domain_moving_mean, domain_moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(training,
                                                lambda: adj_scale,
                                                lambda: array_ops.ones_like(adj_scale))
                adj_bias = tf_utils.smart_cond(training,
                                               lambda: adj_bias,
                                               lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = domain_moving_mean
            moving_variance = domain_moving_variance

            mean = tf_utils.smart_cond(training,
                                       lambda: mean,
                                       lambda: ops.convert_to_tensor(moving_mean))
            variance = tf_utils.smart_cond(
                training,
                lambda: variance,
                lambda: ops.convert_to_tensor(moving_variance))

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                inputs_size = array_ops.size(inputs)
            else:
                inputs_size = None
            if self.renorm:
                domain_moving_stddev = self.moving_stddev[domain_index]

                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    domain_index, new_mean, new_variance, training, inputs_size)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   inputs_size)

            def mean_update():
                true_branch = lambda: _do_update(domain_moving_mean, new_mean)
                false_branch = lambda: domain_moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(domain_moving_stddev,
                                               math_ops.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        domain_moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon))

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    true_branch = lambda: _do_update(domain_moving_variance, new_variance)

                false_branch = lambda: domain_moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        # math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs


def partitioned_normalization(inputs,
                              num_domain,
                              domain_index,
                              axis=-1,
                              momentum=0.99,
                              epsilon=1e-3,
                              center=True,
                              scale=True,
                              beta_initializer=init_ops.zeros_initializer(),
                              gamma_initializer=init_ops.ones_initializer(),
                              moving_mean_initializer=init_ops.zeros_initializer(),
                              moving_variance_initializer=init_ops.ones_initializer(),
                              beta_regularizer=None,
                              gamma_regularizer=None,
                              beta_constraint=None,
                              gamma_constraint=None,
                              training=False,
                              trainable=True,
                              name=None,
                              reuse=None,
                              renorm=False,
                              renorm_clipping=None,
                              renorm_momentum=0.99,
                              fused=None,
                              virtual_batch_size=None,
                              adjustment=None):
    """Functional interface for the batch normalization layer.

    Reference: http://arxiv.org/abs/1502.03167

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    need to be executed alongside the `train_op`. Also, be sure to add any
    batch_normalization ops before getting the update_ops collection. Otherwise,
    update_ops will be empty, and training/inference will not work properly. For
    example:

    ```python
      x_norm = tf.compat.v1.layers.batch_normalization(x, training=training)

      # ...

      update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = optimizer.minimize(loss)
      train_op = tf.group([train_op, update_ops])
    ```

    Arguments:
      inputs: Tensor input.
      axis: An `int`, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: An optional projection function to be applied to the `beta`
          weight after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      gamma_constraint: An optional projection function to be applied to the
          `gamma` weight after being updated by an `Optimizer`.
      training: Either a Python boolean, or a TensorFlow boolean scalar tensor
        (e.g. a placeholder). Whether to return the output in training mode
        (normalized with statistics of the current batch) or in inference mode
        (normalized with moving statistics). **NOTE**: make sure to set this
        parameter correctly, or else your training/inference will not work
        properly.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
      fused: if `None` or `True`, use a faster, fused implementation if possible.
        If `False`, use the system recommended implementation.
      virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
        which means batch normalization is performed across the whole batch. When
        `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        Normalization", which creates virtual sub-batches which are each
        normalized separately (with shared gamma, beta, and moving statistics).
        Must divide the actual batch size during execution.
      adjustment: A function taking the `Tensor` containing the (dynamic) shape of
        the input tensor and returning a pair (scale, bias) to apply to the
        normalized values (before gamma and beta), only during training. For
        example, if axis==-1,
          `adjustment = lambda shape: (
            tf.random.uniform(shape[-1:], 0.93, 1.07),
            tf.random.uniform(shape[-1:], -0.1, 0.1))`
        will scale the normalized value by up to 7% up or down, then shift the
        result by up to 0.1 (with independent scaling and bias for each feature
        but shared across all examples), and finally apply gamma and/or beta. If
        `None`, no adjustment is applied. Cannot be specified if
        virtual_batch_size is specified.

    Returns:
      Output tensor.

    Raises:
      ValueError: if eager execution is enabled.
    """
    layer = PartitionedNormalization(
        num_domain=num_domain,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=name,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs, domain_index, training=training)


class STAR:
    def __init__(self,
                 fields: List[Field],
                 num_domain: int,
                 attention_agg: Type[AttentionBase],
                 star_fcn_input_size: int,
                 star_fcn_units: List[int],
                 aux_net_hidden_units: List[int],
                 aux_net_activation: Callable = dice,
                 aux_net_dropout: Optional[float] = 0.,
                 aux_net_use_bn: Optional[bool] = True,
                 l2_reg: float = 0.,
                 gru_hidden_size: int = 1,
                 attention_hidden_units: List[int] = [80, 40],
                 attention_activation: Callable = tf.nn.sigmoid,
                 domain_indicator_field_name: str = 'domain_indicator',
                 mode: str = 'concat'):
        """

        :param fields: 特征列表
        :param num_domain: 场景数量
        :param attention_agg: Attention聚合, 参考DIN,DIEN
        :param star_fcn_input_size: Star Topology FCN的第一层输入size
        :param star_fcn_units: Star Topology FCN的隐藏层size列表
        :param aux_net_hidden_units: 辅助网络的隐藏层size列表
        :param aux_net_activation: 辅助网络激活函数
        :param aux_net_dropout: 辅助网络dropout
        :param aux_net_use_bn: 辅助网络是否使用BN
        :param l2_reg: 正则惩罚项
        :param gru_hidden_size: Attention参数
        :param attention_hidden_units: Attention参数
        :param attention_activation: Attention参数
        :param domain_indicator_field_name: 场景指示器对应的field名称
        :param mode: item的属性embeddings聚合方式，如mode='concat' 则为`e = [e_{goods_id}, e_{shop_id}, e_{cate_id}]`
        """

        self.domain_indicator_field_name = domain_indicator_field_name

        self.embedding_table = {}
        for field in fields:
            self.embedding_table[field.name] = tf.get_variable(f'{field.name}_embedding_table',
                                                               shape=[field.vocabulary_size, field.dim],
                                                               initializer=tf.truncated_normal_initializer(field.init_mean, field.init_std),
                                                               regularizer=tf.contrib.layers.l2_regularizer(field.l2_reg)
                                                               )

        assert domain_indicator_field_name in self.embedding_table, f"The field of domain indicator is missing: `{domain_indicator_field_name}`"

        mode = mode.lower()
        if mode == 'concat':
            self.func = partial(tf.concat, axis=-1)
        elif mode == 'sum':
            self.func = lambda data: sum(data)
        elif mode == 'mean':
            self.func = lambda data: sum(data) / len(data)
        else:
            raise NotImplementedError(f"`mode` only supports 'mean' or 'concat' or 'sum', but got '{mode}'")

        with tf.variable_scope(name_or_scope='attention_layer'):
            self.attention_agg = attention_agg(gru_hidden_size, attention_hidden_units, attention_activation)

        self.start_pn = partial(partitioned_normalization, num_domain=num_domain, name='star_pn')
        self.aux_net_pn = partial(partitioned_normalization, num_domain=num_domain, name='aux_net_pn')

        with tf.variable_scope(name_or_scope='star_fcn'):
            self.shared_bias = [tf.get_variable(f'star_fcn_b_shared_{i}', shape=[star_fcn_units[i]])
                                for i in range(len(star_fcn_units))]
            self.domain_bias = [tf.get_variable(f'star_fcn_b_domain_{i}', shape=[num_domain, star_fcn_units[i]])
                                for i in range(len(star_fcn_units))]

            star_fcn_units.insert(0, star_fcn_input_size)
            self.shared_weights = [tf.get_variable(f'star_fcn_w_shared_{i}', shape=[star_fcn_units[i], star_fcn_units[i+1]],
                                                   regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   initializer=init_ops.glorot_normal_initializer(), )
                                   for i in range(len(star_fcn_units) - 1)]
            self.domain_weights = [tf.get_variable(f'star_fcn_w_domain_{i}', shape=[num_domain, star_fcn_units[i], star_fcn_units[i + 1]],
                                                   regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   initializer=init_ops.glorot_normal_initializer(), )
                                   for i in range(len(star_fcn_units) - 1)]

        with tf.variable_scope(name_or_scope='auxiliary_network'):
            self.auxiliary_network = partial(dnn_layer,
                                             hidden_units=aux_net_hidden_units,
                                             activation=aux_net_activation,
                                             use_bn=aux_net_use_bn,
                                             dropout=aux_net_dropout,
                                             l2_reg=l2_reg)

    def star_fcn(self, inputs, domain_index, layer_index):
        weights = self.shared_weights[layer_index] * self.domain_weights[layer_index][domain_index]
        bias = self.shared_bias[layer_index] + self.domain_bias[layer_index][domain_index]

        return math_ops.matmul(inputs, weights) + bias

    def __call__(self,
                 user_behaviors_ids: Dict[str, tf.Tensor],
                 sequence_length: tf.Tensor,
                 target_ids: Dict[str, tf.Tensor],
                 other_feature_ids: Dict[str, tf.Tensor],
                 domain_index: tf.Tensor,
                 is_training: bool = True
                 ):
        """

        :param user_behaviors_ids: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id
        :param sequence_length: 用户行为序列长度 [B]
        :param target_ids: 候选ID [B]
        :param other_feature_ids: 其他特征，如用户特征及上下文特征
        :param domain_index: 场景指示器ID，表示当前mini-batch为第n个场景，只取第一个数据来表示当前场景ID
        :param is_training:
        :return:
        """
        # 用户行为历史embedding
        user_behaviors_embeddings = []
        target_embeddings = []
        for name in user_behaviors_ids:
            user_behaviors_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], user_behaviors_ids[name]))
            target_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], target_ids[name]))
        user_behaviors_embeddings = self.func(user_behaviors_embeddings)
        target_embeddings = self.func(target_embeddings)

        # 其他特征embedding
        other_feature_embeddings = []
        for name in other_feature_ids:
            other_feature_embeddings.append(tf.nn.embedding_lookup(self.embedding_table[name], other_feature_ids[name]))
        other_feature_embeddings = array_ops.concat(other_feature_embeddings, axis=-1)

        with tf.variable_scope(name_or_scope='attention_layer'):
            att_outputs = self.attention_agg(user_behaviors_embeddings, target_embeddings, sequence_length)
            if isinstance(att_outputs, (list, tuple)):
                att_outputs = att_outputs[-1]

        domain_index = array_ops.reshape(domain_index, [-1])[0]

        with tf.variable_scope(name_or_scope='partitioned_normalization'):
            agg_inputs = array_ops.concat([att_outputs, target_embeddings, other_feature_embeddings], axis=-1)

            pn_outputs = self.start_pn(agg_inputs, domain_index=domain_index, training=is_training)

        with tf.variable_scope(name_or_scope='star_fcn'):
            star_fcn_outputs = pn_outputs
            for i in range(len(self.shared_weights)):
                star_fcn_outputs = self.star_fcn(star_fcn_outputs, domain_index, i)

            star_logit = tf.layers.dense(star_fcn_outputs, 1, kernel_initializer=init_ops.glorot_normal_initializer())

        with tf.variable_scope(name_or_scope='auxiliary_network'):
            # 场景指示器
            domain_embedding = tf.nn.embedding_lookup(self.embedding_table[self.domain_indicator_field_name], domain_index)
            aux_inputs = array_ops.concat([array_ops.repeat(array_ops.reshape(domain_embedding, [1, -1]),
                                                     array_ops.shape(agg_inputs)[0], axis=0), agg_inputs], axis=-1)
            aux_inputs = self.aux_net_pn(aux_inputs, domain_index=domain_index, training=is_training)
            aux_outputs = self.auxiliary_network(aux_inputs, is_training=is_training)

            aux_logit = tf.layers.dense(aux_outputs, 1, kernel_initializer=init_ops.glorot_normal_initializer())

        return array_ops.reshape(tf.nn.sigmoid(star_logit + aux_logit), [-1])
