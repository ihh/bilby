import glob
import json
import os
import pdb
import sys
import re

from itertools import chain
from natsort import natsorted
import numpy as np
from scipy.sparse import dok_matrix

import tensorflow as tf
import tensorflow_datasets as tfds

import logging

# TFRecord constants
TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'

# Prevent Tensorflow from grabbing GPU memory
tf.config.experimental.set_visible_devices([], "GPU")

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

# target metadata extraction functions
def get_target_type(description):
    match = re.search(r'^(\w+):', description)
    return match.group(1) if match else None

def get_orientation_type(identifier):
    c = identifier[-1]
    if c == '+':
        return 1
    elif c == '-':
        return -1
    return 0

class SeqDataset:
    def __init__(self, data_dir='data', split_label='train', batch_size=2, shuffle_buffer=128, seq_length_crop=None, mode='eval', drop_remainder=False):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seq_length_crop = seq_length_crop
        self.mode = mode
        self.drop_remainder = drop_remainder
        
        data_stats_file = '%s/statistics.json' % self.data_dir
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats['seq_length']
        
        self.seq_depth = data_stats.get('seq_depth',4)
        self.seq_1hot = data_stats.get('seq_1hot', False)
        self.target_length = data_stats['target_length']
        self.num_targets = data_stats['num_targets']
        self.pool_width = data_stats['pool_width']

        self.tfr_path = '%s/tfrecords/%s-*.tfr' % (self.data_dir, self.split_label)
        self.num_seqs = data_stats['%s_seqs' % self.split_label]

        data_targets_file = '%s/targets.txt' % self.data_dir
        with open(data_targets_file) as data_targets_open:
            targets_header, *targets = data_targets_open.read().splitlines()
            id_index, strand_pair_index, description_index = [targets_header.split('\t').index(col) for col in ['identifier', 'strand_pair', 'description']]
            self.strand_pair = [int(line.split('\t')[strand_pair_index]) for line in targets]
            assert len(self.strand_pair) == self.num_targets
            target_type_str = [get_target_type(line.split('\t')[description_index]) for line in targets]
            self.target_type_name = list(set(target_type_str))
            self.target_type = [self.target_type_name.index(t) for t in target_type_str]
            self.orientation_type = [get_orientation_type(line.split('\t')[id_index]) for line in targets]

        self.make_dataset()

    def generate_parser(self, raw=False):
        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            # define features
            features = {
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string),
            }

            # parse example into features
            parsed_features = tf.io.parse_single_example(example_protos, features=features)

            # decode sequence
            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
            if not raw:
                if self.seq_1hot:
                    sequence = tf.reshape(sequence, [self.seq_length])
                    sequence = tf.one_hot(sequence, 1+self.seq_depth, dtype=tf.uint8)
                    sequence = sequence[:,:-1] # drop N
                else:
                    sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
                if self.seq_length_crop is not None:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    sequence = sequence[crop_len:-crop_len,:]
                sequence = tf.cast(sequence, tf.float32)
                
            # decode targets
            targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
            if not raw:
                targets = tf.cast(targets, tf.float32)

            return sequence, targets

        return parse_proto

    def make_dataset(self, cycle_length=4):
        """Make Dataset w/ transformations."""

        # initialize dataset from TFRecords glob
        tfr_files = natsorted(glob.glob(self.tfr_path))
        if tfr_files:
            # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print('Cannot order TFRecords %s' % self.tfr_path, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(self.tfr_path)

        # train
        if self.mode == 'train':
            # repeat
            dataset = dataset.repeat()

            # interleave files
            dataset = dataset.interleave(map_func=file_to_records,
            cycle_length=cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # shuffle
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer,
            reshuffle_each_iteration=True)

        # valid/test
        else:
            # flat mix files
            dataset = dataset.flat_map(file_to_records)

        # (no longer necessary in tf2?)
        # helper for training on single genomes in a multiple genome mode
        # if self.num_seqs > 0:
        #  dataset = dataset.map(self.generate_parser())
        dataset = dataset.map(self.generate_parser())

        # cache (runs OOM)
        # dataset = dataset.cache()

        # batch
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)

        # prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # hold on
        self.dataset = dataset

def targets_prep_strand(targets_df):
    """Adjust targets table for merged stranded datasets.

    Args:
        targets_df: pandas DataFrame of targets

    Returns:
        targets_df: pandas DataFrame of targets, with stranded
            targets collapsed into a single row
    """
    # attach strand
    targets_strand = []
    for _, target in targets_df.iterrows():
        if target.strand_pair == target.name:
            targets_strand.append(".")
        else:
            targets_strand.append(target.identifier[-1])
    targets_df["strand"] = targets_strand

    # collapse stranded
    strand_mask = targets_df.strand != "-"
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df

def make_strand_transform(targets_df, targets_strand_df):
    """Make a sparse matrix to sum strand pairs.

    Args:
        targets_df (pd.DataFrame): Targets DataFrame.
        targets_strand_df (pd.DataFrame): Targets DataFrame, with strand pairs collapsed.

    Returns:
        scipy.sparse.dok_matrix: Sparse matrix to sum strand pairs.
    """

    # initialize sparse matrix
    strand_transform = dok_matrix((targets_df.shape[0], targets_strand_df.shape[0]))

    # fill in matrix
    ti = 0
    sti = 0
    for _, target in targets_df.iterrows():
        strand_transform[ti, sti] = True
        if target.strand_pair == target.name:
            sti += 1
        else:
            if target.identifier[-1] == "-":
                sti += 1
        ti += 1

    return strand_transform

def untransform_preds(preds, targets_df, unscale=False):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # clip soft
    cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
    preds_unclip = cs - 1 + (preds - cs + 1) ** 2
    preds = np.where(preds > cs, preds_unclip, preds)

    # sqrt
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = -1 + (preds[:, sqrt_mask] + 1) ** 2  # (4 / 3)

    # scale
    if unscale:
        scale = np.expand_dims(np.array(targets_df.scale), axis=0)
        preds = preds / scale

    return preds


def untransform_preds1(preds, targets_df, unscale=False):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # scale
    scale = np.expand_dims(np.array(targets_df.scale), axis=0)
    preds = preds / scale

    # clip soft
    cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
    preds_unclip = cs + (preds - cs) ** 2
    preds = np.where(preds > cs, preds_unclip, preds)

    # ** 0.75
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = (preds[:, sqrt_mask]) ** (4 / 3)

    # unscale
    if not unscale:
        preds = preds * scale

    return preds