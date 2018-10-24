# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function

import json
import math
import os
import sys
import time
from datetime import datetime
import numpy as np

import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

print_grad = True


class printOut(object):
  def __init__(self,f=None ,stdout_print=True):
    self.out_file = f
    self.stdout_print = stdout_print

  def print_out(self, s, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
      s = s.decode("utf-8")

    if self.out_file:
      self.out_file.write(s)
      if new_line:
        self.out_file.write("\n")
    self.out_file.flush()

    # stdout
    if self.stdout_print:
      print(s, end="", file=sys.stdout)
      if new_line:
        sys.stdout.write("\n")
      sys.stdout.flush()

  def print_time(self,s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
    return time.time()

  def print_grad(self,model, last=False):
    # gets a model and prints the second norm of the weights and gradients
    if print_grad:
      for tag, value in model.named_parameters():
        if value.grad is not None:
          self.print_out('{0: <50}'.format(tag)+ "\t-- value:" \
            +'%.12f' % value.norm().data[0]+ "\t -- grad: "+ str(value.grad.norm().data[0]))
        else:
          self.print_out('{0: <50}'.format(tag)+ "\t-- value:" +\
            '%.12f' % value.norm().data[0])
      self.print_out("-----------------------------------")
      if last:
        self.print_out("-----------------------------------")
        self.print_out("-----------------------------------")

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def to_np(x):
  return x.data.cpu().numpy()

def to_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)   

#  for extracting the gradients
def extract(xVar):
  global yGrad
  yGrad = xVar
  print(yGrad)

def extract_norm(xVar):
  global yGrad
  yGradNorm = xVar.norm() 
  print(yGradNorm)

# tensorboard logger
class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def _single_cell(unit_type, num_units, forget_bias, dropout, prt,
                                 residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer

    # Cell Type
    if unit_type == "lstm":
        prt.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=forget_bias)
    elif unit_type == "gru":
        prt.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))
        prt.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                                        new_line=False)

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
        prt.print_out("  %s" % type(single_cell).__name__, new_line=False)

    # Device Wrapper
    """ if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        prt.print_out("  %s, device=%s" %
                                        (type(single_cell).__name__, device_str), new_line=False)"""

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
                             forget_bias, dropout, mode, prt, num_gpus, base_gpu=0):
    """Create a list of RNN cells."""
    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        prt.print_out("  cell %d" % i, new_line=False)
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        single_cell = _single_cell(
                unit_type=unit_type,
                num_units=num_units,
                forget_bias=forget_bias,
                dropout=dropout,
                prt=prt,
                residual_connection=(i >= num_layers - num_residual_layers),
                device_str=get_device_str(i + base_gpu, num_gpus),
        )
        prt.print_out("")
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                                        forget_bias, dropout, mode, prt , num_gpus, base_gpu=0):
    """Create multi-layer RNN cell.

    Args:
        unit_type: string representing the unit type, i.e. "lstm".
        num_units: the depth of each unit.
        num_layers: number of cells.
        num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
            cells in the returned list will be wrapped with `ResidualWrapper`.
        forget_bias: the initial forget bias of the RNNCell(s).
        dropout: floating point value between 0.0 and 1.0:
            the probability of dropout.  this is ignored if `mode != TRAIN`.
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER
        num_gpus: The number of gpus to use when performing round-robin
            placement of layers.
        base_gpu: The gpu device id to use for the first RNN cell in the
            returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
            as its device id.

    Returns:
        An `RNNCell` instance.
    """

    cell_list = _cell_list(unit_type=unit_type,
                             num_units=num_units,
                             num_layers=num_layers,
                             num_residual_layers=num_residual_layers,
                             forget_bias=forget_bias,
                             dropout=dropout,
                             mode=mode,
                             prt=prt,
                             num_gpus=num_gpus,
                             base_gpu=base_gpu)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def gradient_clip(gradients, params, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
            tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary

def create_or_load_model(model, model_dir, session, out_dir, name):
    """Create translation model and initialize or load parameters in session."""
    start_time = time.time()
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.saver.restore(session, latest_ckpt)
        utils.print_out(
                "  loaded %s model parameters from %s, time %.2fs" %
                (name, latest_ckpt, time.time() - start_time))
    else:
        utils.print_out("  created %s model with fresh parameters, time %.2fs." %
                                        (name, time.time() - start_time))
        session.run(tf.global_variables_initializer())

    global_step = model.global_step.eval(session=session)
    return model, global_step
    
def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output

def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto

def check_tensorflow_version():
    if tf.__version__ < "1.2.1":
        raise EnvironmentError("Tensorflow version must >= 1.2.1")

def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)

def tf_print(tensor, transform=None):

    # Insert a custom python operation into the graph that does nothing but print a tensors value 
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res
