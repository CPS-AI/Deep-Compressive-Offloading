# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
		Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
		Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
	"""Performs a batch normalization using a standard set of parameters."""
	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide#common_fused_ops
	return tf.layers.batch_normalization(
			inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
			momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
			scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
	"""Pads the input along the spatial dimensions independently of input size.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
								 Should be a positive integer.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		A tensor with the same format as the input with the data either intact
		(if kernel_size == 1) or padded (if kernel_size > 1).
	"""
	pad_total = kernel_size - 1
	pad_beg = pad_total // 2
	pad_end = pad_total - pad_beg

	if data_format == 'channels_first':
		padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
																		[pad_beg, pad_end], [pad_beg, pad_end]])
	else:
		padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
																		[pad_beg, pad_end], [0, 0]])
	return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
	"""Strided 2-D convolution with explicit padding."""
	# The padding is consistent and is based only on `kernel_size`, not on the
	# dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
	if strides > 1:
		inputs = fixed_padding(inputs, kernel_size, data_format)

	return tf.layers.conv2d(
			inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
			padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
			kernel_initializer=tf.variance_scaling_initializer(),
			data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
											 data_format):
	"""A single block for ResNet v1, without a bottleneck.

	Convolution then batch normalization then ReLU as described by:
		Deep Residual Learning for Image Recognition
		https://arxiv.org/pdf/1512.03385.pdf
		by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		filters: The number of filters for the convolutions.
		training: A Boolean for whether the model is in training or inference
			mode. Needed for batch normalization.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: The block's stride. If greater than 1, this block will ultimately
			downsample the input.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		The output tensor of the block; shape should match inputs.
	"""
	shortcut = inputs

	if projection_shortcut is not None:
		shortcut = projection_shortcut(inputs)
		shortcut = batch_norm(inputs=shortcut, training=training,
													data_format=data_format)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides,
			data_format=data_format)
	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=1,
			data_format=data_format)
	inputs = batch_norm(inputs, training, data_format)
	inputs += shortcut
	inputs = tf.nn.relu(inputs)

	return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
											 data_format):
	"""A single block for ResNet v2, without a bottleneck.

	Batch normalization then ReLu then convolution as described by:
		Identity Mappings in Deep Residual Networks
		https://arxiv.org/pdf/1603.05027.pdf
		by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		filters: The number of filters for the convolutions.
		training: A Boolean for whether the model is in training or inference
			mode. Needed for batch normalization.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: The block's stride. If greater than 1, this block will ultimately
			downsample the input.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		The output tensor of the block; shape should match inputs.
	"""
	shortcut = inputs
	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)

	# The projection shortcut should come after the first batch norm and ReLU
	# since it performs a 1x1 convolution.
	if projection_shortcut is not None:
		shortcut = projection_shortcut(inputs)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides,
			data_format=data_format)

	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=1,
			data_format=data_format)

	return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
												 strides, data_format):
	"""A single block for ResNet v1, with a bottleneck.

	Similar to _building_block_v1(), except using the "bottleneck" blocks
	described in:
		Convolution then batch normalization then ReLU as described by:
			Deep Residual Learning for Image Recognition
			https://arxiv.org/pdf/1512.03385.pdf
			by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		filters: The number of filters for the convolutions.
		training: A Boolean for whether the model is in training or inference
			mode. Needed for batch normalization.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: The block's stride. If greater than 1, this block will ultimately
			downsample the input.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		The output tensor of the block; shape should match inputs.
	"""
	shortcut = inputs

	if projection_shortcut is not None:
		shortcut = projection_shortcut(inputs)
		shortcut = batch_norm(inputs=shortcut, training=training,
													data_format=data_format)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=1, strides=1,
			data_format=data_format)
	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides,
			data_format=data_format)
	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
			data_format=data_format)
	inputs = batch_norm(inputs, training, data_format)
	inputs += shortcut
	inputs = tf.nn.relu(inputs)

	return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
												 strides, data_format):
	"""A single block for ResNet v2, with a bottleneck.

	Similar to _building_block_v2(), except using the "bottleneck" blocks
	described in:
		Convolution then batch normalization then ReLU as described by:
			Deep Residual Learning for Image Recognition
			https://arxiv.org/pdf/1512.03385.pdf
			by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

	Adapted to the ordering conventions of:
		Batch normalization then ReLu then convolution as described by:
			Identity Mappings in Deep Residual Networks
			https://arxiv.org/pdf/1603.05027.pdf
			by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		filters: The number of filters for the convolutions.
		training: A Boolean for whether the model is in training or inference
			mode. Needed for batch normalization.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: The block's stride. If greater than 1, this block will ultimately
			downsample the input.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		The output tensor of the block; shape should match inputs.
	"""
	shortcut = inputs
	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)

	# The projection shortcut should come after the first batch norm and ReLU
	# since it performs a 1x1 convolution.
	if projection_shortcut is not None:
		shortcut = projection_shortcut(inputs)

	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=1, strides=1,
			data_format=data_format)

	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides,
			data_format=data_format)

	inputs = batch_norm(inputs, training, data_format)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
			data_format=data_format)

	return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
								training, name, data_format):
	"""Creates one layer of blocks for the ResNet model.

	Args:
		inputs: A tensor of size [batch, channels, height_in, width_in] or
			[batch, height_in, width_in, channels] depending on data_format.
		filters: The number of filters for the first convolution of the layer.
		bottleneck: Is the block created a bottleneck block.
		block_fn: The block to use within the model, either `building_block` or
			`bottleneck_block`.
		blocks: The number of blocks contained in the layer.
		strides: The stride to use for the first convolution of the layer. If
			greater than 1, this layer will ultimately downsample the input.
		training: Either True or False, whether we are currently training the
			model. Needed for batch norm.
		name: A string name for the tensor output of the block layer.
		data_format: The input format ('channels_last' or 'channels_first').

	Returns:
		The output tensor of the block layer.
	"""

	# Bottleneck blocks end with 4x the number of filters as they start with
	filters_out = filters * 4 if bottleneck else filters

	def projection_shortcut(inputs):
		return conv2d_fixed_padding(
				inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
				data_format=data_format)

	# Only the first block per block_layer uses projection_shortcut and strides
	inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
										data_format)

	for _ in range(1, blocks):
		inputs = block_fn(inputs, filters, training, None, 1, data_format)

	return tf.identity(inputs, name)

def orthogonal_regularizer(scale, data_format) :
	""" Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

	def ortho_reg(w) :
		""" Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
		# if data_format == 'channels_first':
		# 	_, c, _, _ = w.get_shape().as_list()
		# else:
		# 	_, _, _, c = w.get_shape().as_list()
		if data_format == 'channels_first':
			w = tf.transpose(w, [0, 2, 3, 1])
		_, _, _, c = w.get_shape().as_list()
		w = tf.reshape(w, [-1, c])

		""" Declaring a Identity Tensor of appropriate size"""
		identity = tf.eye(c)

		""" Regularizer Wt*W - I """
		w_transpose = tf.transpose(w)
		w_mul = tf.matmul(w_transpose, w)
		reg = tf.subtract(w_mul, identity)

		"""Calculating the Loss Obtained"""
		ortho_loss = tf.nn.l2_loss(reg)

		return scale * ortho_loss

	return ortho_reg

def orthogonal_regularizer_fully(scale) :
	def ortho_reg_fully(w) :
		_, c = w.get_shape().as_list()
		identity = tf.eye(c)
		w_transpose = tf.transpose(w)
		w_mul = tf.matmul(w_transpose, w)
		reg = tf.subtract(w_mul, identity)

		ortho_loss = tf.nn.l2_loss(reg)

		return scale * ortho_loss
	return ortho_reg_fully

def spectral_norm(w, iteration=1):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u
	v_hat = None
	for i in range(iteration):
		"""
		power iteration
		Usually iteration = 1 will be enough
		"""

		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = tf.nn.l2_normalize(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = tf.nn.l2_normalize(u_)

	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = w / sigma
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, 
				sn=False, data_format=None, scope='conv_0'):
	with tf.variable_scope(scope):
		if pad > 0:
			h = x.get_shape().as_list()[2]
			if h % stride == 0:
				pad = pad * 2
			else:
				pad = max(kernel - (h % stride), 0)

			pad_top = pad // 2
			pad_bottom = pad - pad_top
			pad_left = pad // 2
			pad_right = pad - pad_left

			if data_format == 'channels_first':
				if pad_type == 'zero' :
					x = tf.pad(x, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]])
				if pad_type == 'reflect' :
					x = tf.pad(x, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]], 
																				  mode='REFLECT')
			else:
				if pad_type == 'zero' :
					x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
				if pad_type == 'reflect' :
					x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
																				  mode='REFLECT')

		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		weight_regularizer = orthogonal_regularizer(0.0001, data_format)
		if sn :
			if data_format == 'channels_first':
				w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[1], channels], 
										initializer=weight_init, regularizer=weight_regularizer)
				x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, 1, stride, stride], padding='VALID', data_format='NCHW')
			else:
				w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], 
										initializer=weight_init, regularizer=weight_regularizer)
				x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID', data_format='NHWC')
			if use_bias :
				bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
				if data_format == 'channels_first':
					x = tf.nn.bias_add(x, bias, data_format='NCHW')
				else:
					x = tf.nn.bias_add(x, bias, data_format='NHWC')
		else:
			x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, 
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, use_bias=use_bias,
                                     data_format=data_format)
	return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, 
				sn=False, data_format=None, scope='deconv_0'):
	with tf.variable_scope(scope):
		x_shape = x.get_shape().as_list()
		if data_format == 'channels_first':
			if padding == 'SAME':
				output_shape = [tf.shape(x)[0], channels, x_shape[2] * stride, x_shape[3] * stride]
			else:
				output_shape =[tf.shape(x)[0], channels, x_shape[2] * stride + max(kernel - stride, 0), 
									x_shape[3] * stride + max(kernel - stride, 0)]
		else:
			if padding == 'SAME':
				output_shape = [tf.shape(x)[0], x_shape[1] * stride, x_shape[2] * stride, channels]
			else:
				output_shape =[tf.shape(x)[0], x_shape[1] * stride + max(kernel - stride, 0), 
									x_shape[2] * stride + max(kernel - stride, 0), channels]
		# print('output_shape', output_shape)
		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		weight_regularizer = orthogonal_regularizer(0.0001, data_format)
		if sn:
			if data_format == 'channels_first':
				w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape().as_list()[1]], 
											initializer=weight_init, regularizer=weight_regularizer)
				x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, 
						strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
			else:
				w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape().as_list()[-1]], 
											initializer=weight_init, regularizer=weight_regularizer)
				x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, 
						strides=[1, stride, stride, 1], padding=padding, data_format='NHWC')
			if use_bias :
				bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
				if data_format == 'channels_first':
					x = tf.nn.bias_add(x, bias, data_format='NCHW')
				else:
					x = tf.nn.bias_add(x, bias, data_format='NHWC')
		else:
			x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
					kernel_size=kernel, 
					kernel_initializer=weight_init, 
					kernel_regularizer=weight_regularizer,
					strides=stride, padding=padding, use_bias=use_bias, 
					data_format=data_format)
		return x

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
	with tf.variable_scope(scope):
		shape = x.get_shape().as_list()
		channels = shape[-1]

		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)
		if sn :
			w = tf.get_variable("kernel", [channels, units], tf.float32, 
				initializer=weight_init, regularizer=weight_regularizer_fully)
			if use_bias :
				bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0))
				x = tf.matmul(x, spectral_norm(w)) + bias
			else:
				x = tf.matmul(x, spectral_norm(w))
		else:
			x = tf.layers.dense(x, units=units, 
							kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer_fully, 
                            use_bias=use_bias)
	return x

def resblock_up(x_init, channels, use_bias=True, is_training=True, 
							sn=False, data_format=None, scope='resblock_up'):
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = batch_norm(x_init, is_training, data_format)
			x = tf.nn.relu(x)
			x = deconv(x, channels, kernel=3, stride=2, 
				use_bias=use_bias, sn=sn, data_format=data_format)
		with tf.variable_scope('res2') :
			x = batch_norm(x, is_training, data_format)
			x = tf.nn.relu(x)
			x = deconv(x, channels, kernel=3, stride=1, 
				use_bias=use_bias, sn=sn, data_format=data_format)
		with tf.variable_scope('skip') :
			x_init = deconv(x_init, channels, kernel=3, stride=2, 
				use_bias=use_bias, sn=sn, data_format=data_format)
	return x + x_init

def hw_flatten(x) :
	x_shape = x.get_shape().as_list()
	return tf.reshape(x, shape=[-1, x_shape[1]*x_shape[2], x_shape[3]])

def self_attention_2(x, channels, sn=False, data_format=None, scope='self_attention'):
	with tf.variable_scope(scope):
		# print('atten_in', x.get_shape())
		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		weight_regularizer = orthogonal_regularizer(0.0001, data_format)
		with tf.variable_scope('f_conv'):
			f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, data_format=data_format) 
			f = tf.layers.max_pooling2d(f, pool_size=8, strides=8, padding='SAME', 
															   data_format=data_format)
			if data_format == 'channels_first':
				f = tf.transpose(f, [0, 2, 3, 1])
			# print('f', f.get_shape(), channels // 8)
		with tf.variable_scope('g_conv'):
			g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, data_format=data_format) 
			if data_format == 'channels_first':
				g = tf.transpose(g, [0, 2, 3, 1])
			# print('g', g.get_shape(), channels // 8)
		with tf.variable_scope('h_conv'):
			h = conv(x, channels // 4, kernel=1, stride=1, sn=sn, data_format=data_format)
			# h = tf.layers.max_pooling2d(h, pool_size=6, strides=6, padding='SAME', 
			# 												   data_format=data_format)
			h = tf.layers.max_pooling2d(h, pool_size=8, strides=8, padding='SAME', 
															   data_format=data_format)
			if data_format == 'channels_first':
				h = tf.transpose(h, [0, 2, 3, 1])
			# print('h', h.get_shape(), channels // 4)

		s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) 
		beta = tf.nn.softmax(s)  # attention map
		o = tf.matmul(beta, hw_flatten(h)) 
		gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
		if data_format == 'channels_first':
			o = tf.transpose(o, [0, 2, 1])

		x_shape = x.get_shape().as_list()
		if data_format == 'channels_first':
			o = tf.reshape(o, shape=[-1, channels//4, x_shape[2], x_shape[3]]) 
		else:
			o = tf.reshape(o, shape=[-1, x_shape[1], x_shape[2], channels//4]) 

		o = conv(o, channels, kernel=1, stride=1, sn=sn, data_format=data_format, scope='attn_conv')
		x = gamma * o + x
		return x

def self_attention_full(x, channels, sn=False, data_format=None, scope='self_attention'):
	with tf.variable_scope(scope):
		# print('atten_in', x.get_shape())
		weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
		weight_regularizer = orthogonal_regularizer(0.0001, data_format)
		with tf.variable_scope('f_conv'):
			f = conv(x, channels, kernel=1, stride=1, sn=sn, data_format=data_format) 
			f = tf.layers.max_pooling2d(f, pool_size=4, strides=4, padding='SAME', 
															   data_format=data_format)
			if data_format == 'channels_first':
				f = tf.transpose(f, [0, 2, 3, 1])
		with tf.variable_scope('g_conv'):
			g = conv(x, channels, kernel=1, stride=1, sn=sn, data_format=data_format) 
			if data_format == 'channels_first':
				g = tf.transpose(g, [0, 2, 3, 1])
		with tf.variable_scope('h_conv'):
			h = conv(x, channels, kernel=1, stride=1, sn=sn, data_format=data_format)
			h = tf.layers.max_pooling2d(h, pool_size=4, strides=4, padding='SAME', 
															   data_format=data_format)
			if data_format == 'channels_first':
				h = tf.transpose(h, [0, 2, 3, 1])

		s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) 
		beta = tf.nn.softmax(s)  # attention map
		o = tf.matmul(beta, hw_flatten(h)) 
		gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
		if data_format == 'channels_first':
			o = tf.transpose(o, [0, 2, 1])

		x_shape = x.get_shape().as_list()
		if data_format == 'channels_first':
			o = tf.reshape(o, shape=[-1, channels, x_shape[2], x_shape[3]]) 
		else:
			o = tf.reshape(o, shape=[-1, x_shape[1], x_shape[2], channels]) 

		o = conv(o, channels, kernel=1, stride=1, sn=sn, data_format=data_format, scope='attn_conv')
		x = gamma * o + x
		return x

class Model(object):
	"""Base class for building the Resnet Model."""

	def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
							 kernel_size,
							 conv_stride, first_pool_size, first_pool_stride,
							 block_sizes, block_strides,
							 resnet_version=DEFAULT_VERSION, data_format=None,
							 dtype=DEFAULT_DTYPE, spectral_norm=False,
							 reuse=False, offload=False, compress_ratio=0.05):
		"""Creates a model for classifying an image.

		Args:
			resnet_size: A single integer for the size of the ResNet model.
			bottleneck: Use regular blocks or bottleneck blocks.
			num_classes: The number of classes used as labels.
			num_filters: The number of filters to use for the first block layer
				of the model. This number is then doubled for each subsequent block
				layer.
			kernel_size: The kernel size to use for convolution.
			conv_stride: stride size for the initial convolutional layer
			first_pool_size: Pool size to be used for the first pooling layer.
				If none, the first pooling layer is skipped.
			first_pool_stride: stride size for the first pooling layer. Not used
				if first_pool_size is None.
			block_sizes: A list containing n values, where n is the number of sets of
				block layers desired. Each value should be the number of blocks in the
				i-th set.
			block_strides: List of integers representing the desired stride size for
				each of the sets of block layers. Should be same length as block_sizes.
			resnet_version: Integer representing which version of the ResNet network
				to use. See README for details. Valid values: [1, 2]
			data_format: Input format ('channels_last', 'channels_first', or None).
				If set to None, the format is dependent on whether a GPU is available.
			dtype: The TensorFlow dtype to use for calculations. If not specified
				tf.float32 is used.

		Raises:
			ValueError: if invalid version is selected.
		"""
		self.resnet_size = resnet_size

		if not data_format:
			data_format = (
					'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

		self.resnet_version = resnet_version
		if resnet_version not in (1, 2):
			raise ValueError(
					'Resnet version should be 1 or 2. See README for citations.')

		self.bottleneck = bottleneck
		if bottleneck:
			if resnet_version == 1:
				self.block_fn = _bottleneck_block_v1
			else:
				self.block_fn = _bottleneck_block_v2
		else:
			if resnet_version == 1:
				self.block_fn = _building_block_v1
			else:
				self.block_fn = _building_block_v2

		if dtype not in ALLOWED_TYPES:
			raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

		self.data_format = data_format
		self.num_classes = num_classes
		self.num_filters = num_filters
		self.kernel_size = kernel_size
		self.conv_stride = conv_stride
		self.first_pool_size = first_pool_size
		self.first_pool_stride = first_pool_stride
		self.block_sizes = block_sizes
		self.block_strides = block_strides
		self.dtype = dtype
		self.pre_activation = resnet_version == 2
		self.sn = spectral_norm
		self.reuse = reuse
		self.offload = offload
		self.compress_ratio = compress_ratio

	def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
													 *args, **kwargs):
		"""Creates variables in fp32, then casts to fp16 if necessary.

		This function is a custom getter. A custom getter is a function with the
		same signature as tf.get_variable, except it has an additional getter
		parameter. Custom getters can be passed as the `custom_getter` parameter of
		tf.variable_scope. Then, tf.get_variable will call the custom getter,
		instead of directly getting a variable itself. This can be used to change
		the types of variables that are retrieved with tf.get_variable.
		The `getter` parameter is the underlying variable getter, that would have
		been called if no custom getter was used. Custom getters typically get a
		variable with `getter`, then modify it in some way.

		This custom getter will create an fp32 variable. If a low precision
		(e.g. float16) variable was requested it will then cast the variable to the
		requested dtype. The reason we do not directly create variables in low
		precision dtypes is that applying small gradients to such variables may
		cause the variable not to change.

		Args:
			getter: The underlying variable getter, that has the same signature as
				tf.get_variable and returns a variable.
			name: The name of the variable to get.
			shape: The shape of the variable to get.
			dtype: The dtype of the variable to get. Note that if this is a low
				precision dtype, the variable will be created as a tf.float32 variable,
				then cast to the appropriate dtype
			*args: Additional arguments to pass unmodified to getter.
			**kwargs: Additional keyword arguments to pass unmodified to getter.

		Returns:
			A variable which is cast to fp16 if necessary.
		"""

		if dtype in CASTABLE_TYPES:
			var = getter(name, shape, tf.float32, *args, **kwargs)
			return tf.cast(var, dtype=dtype, name=name + '_cast')
		else:
			return getter(name, shape, dtype, *args, **kwargs)

	def _model_variable_scope(self):
		"""Returns a variable scope that the model should be created under.

		If self.dtype is a castable type, model variable will be created in fp32
		then cast to self.dtype before being used.

		Returns:
			A variable scope for the model.
		"""

		return tf.variable_scope('resnet_model', reuse=self.reuse,
														 custom_getter=self._custom_dtype_getter)

	def __call__(self, inputs, training):
		"""Add operations to classify a batch of input images.

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Set to True to add operations required only when
				training the classifier.

		Returns:
			A logits Tensor with shape [<batch_size>, self.num_classes].
		"""

		def endecoder(inter_rep):
			with tf.variable_scope('endecoder') as scope:
				axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

				out_size = max(int(3*self.compress_ratio*4*4), 1)
				print('out_size', out_size)

				c_sample = conv(inter_rep, out_size, kernel=4, 
						stride=4, sn=self.sn, use_bias=False, 
						data_format=self.data_format, scope='samp_conv')

				num_centers = 32
				quant_centers = tf.get_variable(
			        'quant_centers', shape=(num_centers,), dtype=tf.float32,
			        initializer=tf.random_uniform_initializer(minval=-16., 
			        								maxval=16))

				print('quant_centers', quant_centers)
				print('c_sample', c_sample)
				quant_dist = tf.square(tf.abs(tf.expand_dims(c_sample, axis=-1) - quant_centers))
				phi_soft = tf.nn.softmax(-1. * quant_dist, dim=-1)
				symbols_hard = tf.argmax(phi_soft, axis=-1)
				phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)
				softout = tf.reduce_sum(phi_soft * quant_centers, -1)
				hardout = tf.reduce_sum(phi_hard * quant_centers, -1)
				
				c_sample_q = softout + tf.stop_gradient(hardout - softout)

				print('phi_soft', phi_soft)
				print('phi_hard', phi_hard)
				print('quant_dist', quant_dist)
				print('softout', softout)
				print('hardout', hardout)
				print('c_sample_q', c_sample_q)

				c_recon = self_attention_full(c_sample_q, channels=out_size, sn=self.sn, 
					data_format=self.data_format, scope='self_attention1')
				c_recon = resblock_up(c_recon, channels=64, use_bias=False, 
					is_training=training, sn=self.sn, 
					data_format=self.data_format, scope='resblock_up_x2')
				c_recon = self_attention_2(c_recon, channels=64, sn=self.sn, 
					data_format=self.data_format, scope='self_attention2')
				c_recon = resblock_up(c_recon, channels=32, use_bias=False, 
					is_training=training, sn=self.sn, 
					data_format=self.data_format, scope='resblock_up_x4')

				c_recon = batch_norm(c_recon, training, self.data_format)
				c_recon = tf.nn.relu(c_recon)
				
				if self.data_format == 'channels_first':
					c_recon = tf.pad(c_recon, tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]]))
				else:
					c_recon = tf.pad(c_recon, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
				c_recon = conv(c_recon, channels=3, kernel=3, stride=1,
					use_bias=False, sn=self.sn, data_format=self.data_format, scope='G_logit')


				c_recon = tf.nn.tanh(c_recon)
				_R_MEAN = 123.68
				_G_MEAN = 116.78
				_B_MEAN = 103.94
				_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
				if self.data_format == 'channels_first':
					ch_means = tf.expand_dims(tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 2), 3)
				else:
					ch_means = tf.expand_dims(tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 0), 0)

				return c_sample, (c_recon+1.0)*127.5-ch_means

		@tf.custom_gradient
		def grad1pass(x):
			def grad(dy):
				d_norm = tf.sqrt(tf.reduce_sum(dy*dy))
				return dy*1.0/tf.maximum(1.0, d_norm)
			return x, grad

		with self._model_variable_scope():
			if self.data_format == 'channels_first':
				# Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
				# This provides a large performance boost on GPU. See
				# https://www.tensorflow.org/performance/performance_guide#data_formats
				inputs = tf.transpose(inputs, [0, 3, 1, 2])

			if self.offload:
				c_sample, c_recon = endecoder(inputs)
				print('c_sample', c_sample.get_shape())
				print('c_recon', c_recon.get_shape())

				inputs = grad1pass(c_recon)
				# inputs = c_recon

			inter_feature = []

			inputs = conv2d_fixed_padding(
					inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
					strides=self.conv_stride, data_format=self.data_format)

			# inter_feature.append(inputs)

			inputs = tf.identity(inputs, 'initial_conv')

			# We do not include batch normalization or activation functions in V2
			# for the initial conv1 because the first ResNet unit will perform these
			# for both the shortcut and non-shortcut paths as part of the first
			# block's projection. Cf. Appendix of [2].
			if self.resnet_version == 1:
				inputs = batch_norm(inputs, training, self.data_format)
				inputs = tf.nn.relu(inputs)

			if self.first_pool_size:
				inputs = tf.layers.max_pooling2d(
						inputs=inputs, pool_size=self.first_pool_size,
						strides=self.first_pool_stride, padding='SAME',
						data_format=self.data_format)
				inputs = tf.identity(inputs, 'initial_max_pool')

			for i, num_blocks in enumerate(self.block_sizes):
				num_filters = self.num_filters * (2**i)
				inputs = block_layer(
						inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
						block_fn=self.block_fn, blocks=num_blocks,
						strides=self.block_strides[i], training=training,
						name='block_layer{}'.format(i + 1), data_format=self.data_format)
				if i == 1:
					inter_feature.append(inputs)

			# Only apply the BN and ReLU for model that does pre_activation in each
			# building/bottleneck block, eg resnet V2.
			if self.pre_activation:
				inputs = batch_norm(inputs, training, self.data_format)
				inputs = tf.nn.relu(inputs)

			# The current top layer has shape
			# `batch_size x pool_size x pool_size x final_size`.
			# ResNet does an Average Pooling layer over pool_size,
			# but that is the same as doing a reduce_mean. We do a reduce_mean
			# here because it performs better than AveragePooling2D.
			axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
			inputs = tf.reduce_mean(inputs, axes, keepdims=True)
			inputs = tf.identity(inputs, 'final_reduce_mean')

			inputs = tf.squeeze(inputs, axes)
			inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
			inputs = tf.identity(inputs, 'final_dense')
		if self.offload:
			if self.data_format == 'channels_first':
				c_recon = tf.transpose(c_recon, [0, 2, 3, 1])
			return inputs, c_recon, inter_feature
		else:
			return inputs, inter_feature
