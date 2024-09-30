#!/usr/bin/env python
# Copyright 2023 Calio LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import pdb
import os
import sys
import json
import pickle

import tempfile
import shutil
import tensorflow as tf

sys.path.append (os.path.dirname(__file__) + '/..')
from snps import score_snps

# Prevent Tensorflow from grabbing GPU memory
tf.config.experimental.set_visible_devices([], "GPU")

"""
hound_snp.py

Compute variant effect predictions for SNPs in a VCF file.
"""


def main():
    usage = "usage: %prog [options] <config_file> <params_file> <vcf_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-c",
        dest="cluster_snps_pct",
        default=0,
        type="float",
        help="Cluster SNPs within a %% of the seq length to make a single ref pred [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA [Default: %default]",
    )
    parser.add_option(
        "--indel_stitch",
        dest="indel_stitch",
        default=False,
        action="store_true",
        help="Stitch indel compensation shifts [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="snp_out",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="snp_stats",
        default="logSUM,logD2",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    parser.add_option(
        "--require_gpu",
        dest="require_gpu",
        default=False,
        action="store_true",
        help="Only run on GPU",
    )
    (options, args) = parser.parse_args()

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if len(args) == 3:
        config_file = args[0]
        vars_file = args[1]
        vcf_file = args[2]
    
    elif len(args) == 4:
        # multi separate
        options_pkl_file = args[0]
        config_file = args[1]
        vars_file = args[2]
        vcf_file = args[3]

        # save out dir
        out_dir = options.out_dir

        # load options
        options = load_extra_options(options_pkl_file, options)
        # update output directory
        options.out_dir = out_dir

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        config_file = args[1]
        vars_file = args[2]
        vcf_file = args[3]
        worker_index = int(args[4])

        # load options
        options = load_extra_options(options_pkl_file, options)
        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)
        
    else:
        parser.error("Must provide config, parameters, and QTL VCF files")

    with open(config_file, "r") as fp:
        config = json.load(fp)
    
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.snp_stats = options.snp_stats.split(",")
    if options.targets_file is None:
        parser.error("Must provide targets file")

    #################################################################
    # check if the program is run on GPU, else quit
    physical_devices = tf.config.list_physical_devices()
    # Check if a GPU is available
    gpu_available = any(device.device_type == "GPU" for device in physical_devices)

    if gpu_available:
        print("Running on GPU")
    else:
        print("Running on CPU")
        if options.require_gpu:
            raise SystemExit("Job terminated because it's running on CPU")

    #################################################################
    # calculate SAD scores:
    if options.processes is not None:
        score_snps(config, vars_file, vcf_file, worker_index, options)
    else:
        score_snps(config, vars_file, vcf_file, 0, options)


def load_extra_options(options_pkl_file, options):
    """
    Args:
        options_pkl_file: option file
        options: existing options from command line
    Returns:
        options: updated options
    """
    options_pkl = open(options_pkl_file, "rb")
    new_options = pickle.load(options_pkl)
    new_option_attrs = vars(new_options)
    # Assuming 'options' is the existing options object
    # Update the existing options with the new attributes
    for attr_name, attr_value in new_option_attrs.items():
        setattr(options, attr_name, attr_value)
    options_pkl.close()
    return options

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()