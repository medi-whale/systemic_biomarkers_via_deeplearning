import argparse
import tensorflow as tf, pandas as pd, numpy as np
from progressbar import ProgressBar

# Import custom libs
from utils import TFObjectWrapper

_SHUFFLE_BUFFER_SIZE = 1000 # Buffer size for shuffling training data
_NUMERIC_FEATURES = ['Age', 'gender', 'bmi', 'ldl', 'glucose', 'sbp', 'dbp']

# 1. Encode a feature as tf.train.Feature compatible with tf.train.Example
def _bytes_feature(value):
    """ Returns a bytes_list from a string or byte """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """ Returns a float_list from a float or double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Returns a int64_list from a bool / enum / int / uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Some auxiliary functions to encode features as tf.train.Feature
def _validate_str(value):
    """ Convert a string to a bytes string """
    if not isinstance(value, str):
        value = str(value)
    return str.encode(value, encoding='utf-8')

def feature_description(feature):
    """ Returns feature_description for decode .tfrecords file """
    if feature == "Gender" :
        return {
                'patient_id' : tf.io.FixedLenFeature([], tf.string),
                'date' : tf.io.FixedLenFeature([], tf.string),
                feature : tf.io.FixedLenFeature([], tf.string),
                'fundus' : tf.io.FixedLenFeature([], tf.string),
            }
    else :
        return {
                'patient_id' : tf.io.FixedLenFeature([], tf.string),
                'date' : tf.io.FixedLenFeature([], tf.string),
                feature : tf.io.FixedLenFeature([], tf.float32),
                'fundus' : tf.io.FixedLenFeature([], tf.string),
            }


def _standardize(df, train_stats):
    """ Standardize dataframe using train_stats dictionary
    Args:
        df: A pandas.DataFrame object. Dataset to be standardized
        train_stats: A dictionary of train set statistics. {'feature_name' : {'mean' : mean, 'std' : std}}
    Returns:
        df_stand: A pandas.DataFrame object. A standardized dataset.
    """
    df_stand = df.copy()
    for feature in train_stats.keys():
        stats = train_stats[feature]
        df_stand.loc[:, feature] = (df_stand[feature].values - stats['mean'])/stats['std']
    return df_stand


class Dataset(TFObjectWrapper):
    """ Wrapper class for tf.data.Dataset with numerical data and fundus images """

    def __init__(self,
                 tfrecord_path,
                 batch_size,
                 img_size,
                 crop,
                 flip,
                 rotation,
                 brightness,
                 saturation,
                 training,feature):
        """ Initialize a CombinedDataset instance
        Args:
            tfrecord_path: String, path to csv path
            batch_size: Integer, the number of samples in a batch
            img_size: int, width and height of the image.
            crop: Boolean, whether to crop
            flip: Boolean, whether to flip left or right randomly
            rotation: Boolean, whether to rotate randomly
            brightness: Boolean, whether to adjust brightness randomly
            saturation: Boolean, whether to adjust saturation randomly
            training: Boolean, whether this dataset to be used for training
        Returns:
            A CombinedDataset instance
        """
        # Call super class constructor
        super().__init__()

        # Define attributes
        self.tfrecord_path = tfrecord_path
        self.feature= feature
        self.description = feature_description(self.feature)
        self.batch_size = batch_size
        self.training = training
        self.img_size = img_size
        self.crop = crop
        self.flip = flip
        self.rotation = rotation
        self.brightness = brightness
        self.saturation = saturation
        

        # Build tf.data.Dataset from .tfrecords file
        self.dataset = self._build_dataset()

        # Create iterator depending on training
        if self.training:
            self.iterator = self.dataset.make_one_shot_iterator()
        else:
            self.iterator = self.dataset.make_initializable_iterator()

    def initialize(self):
        if self.training:
            raise ValueError('This dataset cannot be initialized')
        else:
            sess = self.get_current_session()
            sess.run(self.iterator.initializer)

    def get_next(self):
        """ Override get_next() """
        return self.iterator.get_next()

    def _build_dataset(self):
        """ Build a tf.data.TFRecordDataset
        Returns:
            dataset: tf.data.TFRecordDataset
        """
        # Define image size
        if self.crop:
            crop_size = self.img_size
            img_size = int((1.0/0.9)*self.img_size)
        else:
            img_size = self.img_size

        def _parse_fn(example_proto):
            """ Parsing function for a .tfrecords file
            Args:
                example_proto: Serialized tf.train.Example
            Returns:
                parsed example
            """
            parsed_example = tf.io.parse_single_example(example_proto, self.description)
            parsed_example['fundus'] = tf.image.decode_jpeg(parsed_example['fundus'], channels=3)
            return parsed_example

        def _process_fn_train(example):
            """ Apply some random transformations for data augmentation """

            # Random crop
            if self.crop:
                example['fundus'] = tf.image.random_crop(example['fundus'],
                        size=[crop_size, crop_size, 3])
            # Random flip
            if self.flip:
                example['fundus'] = tf.image.random_flip_left_right(example['fundus'])
            # Random rotation
            if self.rotation:
                degree_1 = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
                degree_2 = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
                example['fundus'] = tf.image.rot90(example['fundus'], k=degree_1)
            # Random brightness
            if self.brightness:
                example['fundus'] = tf.image.random_brightness(example['fundus'], max_delta=0.2)
            # Random saturation
            if self.saturation:
                example['fundus'] = tf.image.random_saturation(example['fundus'], lower=0.5, upper=1.5)
            return example

        def _process_fn_eval(example):
            """ Image process function for evaluation which is an argument for tf.data.Dataset.map function """
            # Resize images 

            # Center crop for evaluation
            if self.crop:
                margin = img_size - crop_size
                offset_center = int(margin/2.0)
                example["fundus"] = tf.image.crop_to_bounding_box(example["fundus"], offset_height=offset_center,
                        offset_width=offset_center, target_height=crop_size, target_width=crop_size)

            return example

        def _normalize_fn(example):
            """ Normalize images for training neural nets """
            # Type cast from uint8 to float32 and saturate
            example['fundus'] = tf.image.convert_image_dtype(example['fundus'],
                    dtype=tf.float32, saturate=True)

            # Rescale image to range from -1 to 1
            example['fundus'] = 2*(example['fundus']) - 1
            return example

        def _output_fn(example):
            """ output function to determine example data structure \
                    provided by tf.data.Dataset
            """
            
            features = {'fundus' : example['fundus']}
            labels = {self.feature : example[self.feature],"patient_id":example["patient_id"],"date":example["date"]}

            new_example = (features, labels)
            return new_example

        def _parse_and_process_fn(example):
            """ Which is an argument for tf.data.DataSet.map function \
                    after .example() transformation """
            example = _parse_fn(example)

            if self.training:
                example = _process_fn_train(example)
            else:
                example = _process_fn_eval(example)
            example = _normalize_fn(example)
            example = _output_fn(example)
            return example

        # Read data using tf.data.TFRecordDataset
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        if self.training:
            dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)
            dataset = dataset.repeat()
        dataset = dataset.map(_parse_and_process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset