import os
import json
from itertools import chain
import numpy as np
import tensorflow as tf
from natsort import natsorted
from glob import glob
import logging
from jsonargparse import CLI

# Prevent Tensorflow from grabbing GPU memory
tf.config.experimental.set_visible_devices([], "GPU")

# TFRecord constants
TFR_INPUT = 'sequence'
TFR_OUTPUT = 'target'
TFR_JUNCTION_COORDS = 'junction_coords'
TFR_JUNCTION_COUNTS = 'junction_counts'
TFR_WINDOW_ID = 'window_id'

def concatenate (list_of_lists):
    return list(chain(*list_of_lists))

# Soft-clipping described in Borzoi paper as "squashing"
def squash_clip (x, power=.75, threshold=384, residual_power=0.5):
    x = x ** power
    return np.where (x > threshold, threshold + np.abs(x - threshold) ** residual_power, x)

# Sqrt-clipping
def sqrt_clip (x):
    return np.sqrt(1+x) - 1

def feature_bytes(values):
    """Convert numpy arrays to bytes features."""
    values = values.flatten().tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def build_window_lookup_index (bed_filename, windows_per_tfrecord):
    split_count = {}
    window_id = []
    with open (bed_filename, 'r') as bed_file:
        for line in bed_file:
            id = line.strip().split('\t')[3]
            count = split_count.get(id,0)
            split_count[id] = count + 1
            window_id.append ("%s-%d-%d" % (id, count // windows_per_tfrecord, count % windows_per_tfrecord))
    return dict(zip(window_id, range(len(window_id))))

def add_xxj_to_tfr (tfr_input_filename, tfr_output_filename, intron_counts, window_lookup, clip_mode):
    prefix = tfr_input_filename.split('/')[-1].replace('.tfr', '')
    dataset = tf.data.TFRecordDataset([tfr_input_filename], compression_type='ZLIB')
    features = { TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                 TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string) }
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    with tf.io.TFRecordWriter (tfr_output_filename, tf_opts) as writer:
        for n_record, record in enumerate(dataset):
            parsed_features = tf.io.parse_single_example(record, features=features)
            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8).numpy()
            targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16).numpy()
            window_id = '%s-%d' % (prefix, n_record)
            n_window = window_lookup.get(window_id)
            assert n_window is not None, f"Window {window_id} not found in lookup"
            xxj_data = intron_counts['windows'].get(window_id)
            xxj_coords = np.array (concatenate ([[[n_track, track[0][dac[0]], track[1][dac[1]]] for dac in track[2]] for n_track, track in enumerate(xxj_data)]), dtype=np.uint16)
            xxj_counts = np.array (concatenate ([[dac[2] for dac in track[2]] for track in xxj_data]), dtype=np.float32)
            if clip_mode == "squash":
                xxj_counts = squash_clip (xxj_counts)
            elif clip_mode == "sqrt":
                xxj_counts = sqrt_clip (xxj_counts)
            elif clip_mode == "none":
                pass
            else:
                raise ValueError ("Invalid clip_mode")
            xxj_counts = np.float16 (xxj_counts)
            out_features = { TFR_INPUT: feature_bytes(sequence),
                             TFR_OUTPUT: feature_bytes(targets),
                             TFR_JUNCTION_COORDS: feature_bytes(xxj_coords),
                             TFR_JUNCTION_COUNTS: feature_bytes(xxj_counts),
                             TFR_WINDOW_ID: feature_bytes(np.array([n_window],dtype=np.int32)) }
            example = tf.train.Example(features=tf.train.Features(feature=out_features))
            writer.write(example.SerializeToString())

def main(tfr_input_dir: str,
         tfr_output_dir: str = None,
         tfr_filename: str = None,
         bed_filename: str = None,
         intron_counts_filename: str = None,
         inspect: bool = False,
         inspect_xxj: bool = False,
         inspect_batch: int = None,
         inspect_bin: int = None,
         inspect_track: int = None,
         inspect_xxj_track: int = None,
         windows_per_tfrecord: int = 32,
         stats_filename: str = None,
         clip_mode: str = "sqrt",  # "squash", "sqrt" or "none"
         ):
    tfr_path = '%s/fold*-*.tfr' % tfr_input_dir
    tfr_input_filenames = [tfr_filename] if tfr_filename else natsorted(glob(tfr_path))
    
    inspect_xxj = inspect_xxj or (inspect_xxj_track is not None)
    inspect = inspect or inspect_xxj or tfr_filename or (inspect_batch is not None) or (inspect_bin is not None) or (inspect_track is not None)

    if stats_filename:
        with open(stats_filename, 'r') as f:
            data_stats = json.load(f)
        seq_depth = data_stats.get('seq_depth',4)
        target_length = data_stats['target_length']
        num_targets = data_stats['num_targets']
        pool_width = data_stats['pool_width']
    else:
        seq_depth = 4
        target_length = 8192
        num_targets = 2194
        pool_width = 32

    if inspect:
        dataset = tf.data.TFRecordDataset([tfr_input_filenames], compression_type='ZLIB')
        features = { TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                     TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string) }
        if inspect_xxj:
            features[TFR_JUNCTION_COORDS] = tf.io.FixedLenFeature([], tf.string)
            features[TFR_JUNCTION_COUNTS] = tf.io.FixedLenFeature([], tf.string)
            features[TFR_WINDOW_ID] = tf.io.FixedLenFeature([], tf.string)
        for n, record in enumerate(dataset):
            if inspect_batch is not None and n != inspect_batch:
                continue
            parsed_features = tf.io.parse_single_example(record, features=features)
            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
            targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
            sequence = tf.reshape(sequence, (target_length, seq_depth)).numpy()
            targets = tf.reshape(targets, (num_targets,)).numpy()
            if inspect_bin is not None:
                sequence = sequence[inspect_bin*pool_width:(inspect_bin+1)*pool_width,:]
                targets = targets[:,inspect_bin]
            if inspect_track is not None:
                targets = targets[inspect_track,:]
            print("Sequence %d:" % n, sequence)
            print("Target %d:" % n, targets)
            if inspect_xxj:
                xxj_coords = tf.io.decode_raw(parsed_features[TFR_JUNCTION_COORDS], tf.uint16)
                xxj_counts = tf.io.decode_raw(parsed_features[TFR_JUNCTION_COUNTS], tf.float16)
                window_id = tf.io.decode_raw(parsed_features[TFR_WINDOW_ID], tf.int32)
                xxj_coords = tf.reshape(xxj_coords, (-1, 3)).numpy()
                xxj_counts = xxj_counts.numpy()
                if inspect_xxj_track is not None:
                    xxj_counts = xxj_counts[xxj_coords[:,0] == inspect_xxj_track]
                    xxj_coords = xxj_coords[xxj_coords[:,0] == inspect_xxj_track]
                if inspect_bin is not None:
                    xxj_counts = xxj_counts[xxj_coords[:,1] == inspect_bin]
                    xxj_counts = xxj_counts[xxj_coords[:,1] == inspect_bin]
                print("xxj_coords %d:" % n, xxj_coords)
                print("xxj_counts %d:" % n, xxj_counts)
                print("window_id %d:" % n, window_id)
    else:
        assert tfr_output_dir is not None, "Output directory required"
        assert bed_filename is not None, "BED filename required"
        assert intron_counts_filename is not None, "Intron counts filename required"
        logging.warning("Building window lookup index from %s" % bed_filename)
        window_lookup = build_window_lookup_index(bed_filename, windows_per_tfrecord)
        logging.warning(f"Reading intron counts from {intron_counts_filename}")
        with open(intron_counts_filename, 'r') as f:
            intron_counts = json.load(f)
        for tfr_input_filename in tfr_input_filenames:
            tfr_output_filename = '%s/%s' % (tfr_output_dir, os.path.basename(tfr_input_filename))
            logging.warning(f"Processing {tfr_input_filename} to {tfr_output_filename}")
            add_xxj_to_tfr(tfr_input_filename, tfr_output_filename, intron_counts, window_lookup, clip_mode)

if __name__ == '__main__':
    CLI(main, parser_mode='jsonnet')
