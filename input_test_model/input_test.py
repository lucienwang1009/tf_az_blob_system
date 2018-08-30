import os
import tensorflow as tf
import time
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_NUM_TRAIN_FILES = 1024
_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

def get_filenames(data_dir):
  """Return filenames for dataset."""
  return [
      os.path.join(data_dir, 'train-%05d-of-01024' % i)
      for i in range(_NUM_TRAIN_FILES)]

def process_record_dataset(dataset, batch_size,
                           num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, prefetch=True):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  if prefetch:
    dataset = dataset.prefetch(buffer_size=batch_size)
  # if is_training:
  #   # Shuffle the records. Note that we shuffle before repeating to ensure
  #   # that the shuffling respects epoch boundaries.
  #   dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  # dataset = dataset.apply(
  #     tf.contrib.data.map_and_batch(
  #         lambda value: parse_record_fn(value),
  #         batch_size=batch_size,
  #         num_parallel_batches=1,
  #         drop_remainder=False))
  dataset = dataset.batch(batch_size=batch_size)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  if prefetch:
    # dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=batch_size)

  return dataset

def input_fn(data_dir, batch_size, num_epochs=1, num_gpus=None, prefetch=True):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  # # Shuffle the input files
  # dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  # dataset = dataset.apply(tf.contrib.data.interleave(
  #     tf.data.TFRecordDataset, cycle_length=6))
  dataset = dataset.apply(tf.data.TFRecordDataset)

  return process_record_dataset(
      dataset=dataset,
      batch_size=batch_size,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'],
      prefetch=prefetch
  )

def read_gfile(file_name):
    with tf.gfile.GFile(file_name, 'r') as f:
        f.read()

def gfile_benchmark(data_dir, num_files):
    total_time = 0
    file_names = get_filenames(data_dir)
    for file_name in file_names[:num_files]:
        start_time = time.time()
        read_gfile(file_name)
        total_time += (time.time() - start_time)
    logger.info('Total read files per second: %s' % (num_files / total_time))


def main(data_dir, batch_size, num_steps,
         num_epochs=1,
         num_gpus=None,
         prefetch=True,
         warmup_steps=100):
    dataset = input_fn(data_dir, batch_size, num_epochs, num_gpus, prefetch)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    total_time = 0
    for i in range(num_steps):
        try:
            start_time = time.time()
            sess.run(next_element)
            total_time += (time.time() - start_time)
        except tf.errors.OutOfRangeError:
            break
    logger.info('Total read images per second: %s' % (num_steps * batch_size / total_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read file benchmark for Tensorflow')
    parser.add_argument('benchmark', type=str,
                        help='which benchmark you want to run: gfile or dataset')
    parser.add_argument('--data_dir', type=str,
                        help='the path or url of dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='the size of each batch')
    parser.add_argument('--num_steps', type=int, default=2000,
                        help='for gfile benchmark, it is the number of read files.'
                        'otherwise, it is the number of running batches')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='not used for gfile benchmark')
    # parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--prefetch', action='store_true',
                        help='if use prefetch')

    args = parser.parse_args()
    if args.benchmark == 'gfile':
        gfile_benchmark(args.data_dir, args.num_steps)
    else:
        main(args.data_dir,
             args.batch_size,
             args.num_steps,
             num_epochs=args.batch_size * args.num_steps // _NUM_IMAGES['train'],
             prefetch=args.prefetch,
             warmup_steps=args.warmup_steps)
