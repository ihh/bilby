import os
import math
from itertools import chain
from collections import Counter
import logging
import json
import numpy as np

from jsonargparse import CLI

signs = [+1, -1]
strand_chars = ['+', '-']

def main(bed_filename: str,
         dir_name: str,
         counts_filename: str = 'intron_counts.json',
         out_filename: str = 'processed_intron_counts.json',
         counts_threshold: float = 1,
         bin_size: int = 32,
         pool_strategy: str = 'max',
         windows_per_tfrecord: int = 32,
         ):

    """
    Preprocess exon-exon junction counts that have been mapped to training windows

    Args:
        bed_filename: BED file containing window coordinates and fold split
        windows_per_tfrecord: Number of windows in each TFRecord
        dir_name: Root directory. Subdirectories should contain JSON files with exon-exon junction counts for each window
        counts_filename: name of JSON input file in each subdirectory containing exon-exon junction counts for each window
        out_filename: JSON output file containing processed exon-exon junction counts
        counts_threshold: Minimum count threshold for junctions to be included in the output
        bin_size: Size of the bins to group junctions into
        pool_strategy: Strategy for pooling junction counts that fall into the same bin ('max', 'average', 'sum', 'logsumexp')
    """

    split_count = {}
    window_id = []
    with open (bed_filename, 'r') as bed_file:
        for line in bed_file:
            id = line.strip().split('\t')[3]
            count = split_count.get(id,0)
            split_count[id] = count + 1
            window_id.append ("%s-%d-%d" % (id, count // windows_per_tfrecord, count % windows_per_tfrecord))

    paths = list(t[0] for t in os.walk(dir_name) if t[2].count(counts_filename))
    processed_counts = [process_counts_file(path,counts_filename,counts_threshold,bin_size,pool_strategy) for path in paths]  # (n_tissues, n_windows, 2, ...)
    n_tissues = len(processed_counts)
    n_windows = len(processed_counts[0])
    assert n_tissues == len(paths)
    assert len([pc for pc in processed_counts if len(pc) != n_windows]) == 0
    processed_counts = [concatenate([pc[w] for pc in processed_counts]) for w in range(n_windows)]  # (n_windows, 2*n_tissues, ...)
    n_strands = len(processed_counts[0])
    assert n_strands == 2*n_tissues
    strand_pair = [sp ^ 1 for sp in range(n_strands)]
    source = [os.path.basename(path) + strand for path in paths for strand in strand_chars]
    stats = label_stats ([compute_stats(concatenate([get_lengths(*window[s]) for window in processed_counts])) for s in range(n_strands)])
    global_stats = label_global_stats (compute_stats(concatenate([concatenate([get_lengths(*strand) for strand in window]) for window in processed_counts])))
    result = { 'windows': dict(zip(window_id,processed_counts)), 'strand_pair': strand_pair, 'source': source, 'track_stats': stats, 'all_track_stats': global_stats }
    
    with open ('%s/%s' % (dir_name,out_filename), 'w') as out_file:
        json.dump (result, out_file)

def concatenate (list_of_lists):
    return list(chain(*list_of_lists))

def sgn(x):
    return math.copysign(1,x)

def process_counts_file (path, counts_filename, counts_threshold, bin_size, pool_strategy):
    with open ('%s/%s' % (path, counts_filename), 'r') as in_file:
        intron_counts = json.load (in_file)
        logging.warning(f"Loaded {path}")
    filtered_intron_counts = [[t for t in tuples if t[2] >= counts_threshold] for tuples in intron_counts]
    stranded_intron_counts = [[[t for t in tuples if sgn(t[1]-t[0]) == s] for s in signs] for tuples in filtered_intron_counts]
    return [[process_strand(sic[i],signs[i],bin_size,pool_strategy) for i in [0,1]] for sic in stranded_intron_counts]

def get_bins (sites, bin_size):
    uniq_sites = sorted(set(sites))
    bins = [s//bin_size for s in uniq_sites]
    bin_count = Counter(bins)
    uniq_bins = sorted(bin_count.keys())
    site_weight = {s: 1/bin_count[bins[i]] for i, s in enumerate(uniq_sites)}
    site_to_bin_idx = {s: uniq_bins.index(bins[i]) for i, s in enumerate(uniq_sites)}
    return uniq_bins, site_weight, site_to_bin_idx

def process_strand (intron_counts, _sign, bin_size, pool_strategy):
    donors, acceptors = [sorted(set([t[i] for t in intron_counts])) for i in [0,1]]
    donor_bins, donor_weight, donor_bin_idx = get_bins (donors, bin_size)
    acceptor_bins, acceptor_weight, acceptor_bin_idx = get_bins (acceptors, bin_size)
    weight = dict()
    for (d,a,c) in intron_counts:
        k = str(donor_bin_idx[d]) + ' ' + str(acceptor_bin_idx[a])
        if pool_strategy == 'max':
            weight[k] = max (weight.get(k,0), c)
        elif pool_strategy == 'average':
            w = donor_weight[d] * acceptor_weight[a] * c
            weight[k] = weight.get(k,0) + w
        elif pool_strategy == 'sum':
            weight[k] = weight.get(k,0) + c
        elif pool_strategy == 'logsumexp':
            weight[k] = math.log (math.exp(weight.get(k,float('-inf'))) + math.exp(c))
        else:
            raise ValueError (f"Unknown pooling strategy {pool_strategy}")
    counts = sorted ([tuple([int(x) for x in kv.split()] + [c]) for kv,c in weight.items()])
    return [donor_bins, acceptor_bins, counts]

def get_lengths (donors, acceptors, counts):
    return [1 + abs(acceptors[t[1]] - donors[t[0]]) for t in counts]

def compute_stats (x):
    # return count, min, max, mean, std, 1st/5th/95th/99th percentiles
    return [round(y) for y in [len(x), np.min(x), np.max(x), np.mean(x), np.std(x), np.percentile(x,1), np.percentile(x,5), np.percentile(x,95), np.percentile(x,99)]]

stats_labels = ['count', 'min', 'max', 'mean', 'std', '1%', '5%', '95%', '99%']

def label_stats (stats):
    return { label: [s[i] for s in stats] for i, label in enumerate(stats_labels) }

def label_global_stats (stats):
    return { label: stats[i] for i, label in enumerate(stats_labels) }


if __name__ == "__main__":
    CLI(main, parser_mode='jsonnet')