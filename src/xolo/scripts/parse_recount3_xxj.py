import sys
import os
import re
import gzip
import logging
import json
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import pybedtools

from jsonargparse import CLI

def main(bed_filename: str,
         csv_filename: str,
         flank_len: int = 0,
         weighted_average: bool = False,
         counts_filename: str = 'intron_counts.json',
         ):

    """
    Map exon-exon junction counts from RECOUNT3 to windows in a BED file

    Args:
        bed_filename: BED file containing windows to map junctions to
        csv_filename: CSV file from RECOUNT3 containing URLs for junction count matrices and junction coords
        flank_len: Number of bases to include on either side of each window
        weighted_average: Whether to weight junction counts by total sample counts
        counts_filename: JSON output file containing intron counts for each window
    """

    # Iterate through CSV, downloading files for GTEx datasets
    csv = pd.read_csv(csv_filename)
    for n, row in csv.iterrows():
        if row['file_source'] == 'gtex':
            project = row['project'].lower()
            if not os.path.isdir(project):
                os.mkdir(project)
            out_filename = project + '/' + counts_filename
            if os.path.isfile(out_filename):
                logging.warning(f"{out_filename} is present; skipping")
            else:
                logging.warning(f"Preparing {out_filename}")
                mm_filename = download_file_if_absent (row['jxn_MM'], project)
                rr_filename = download_file_if_absent (row['jxn_RR'], project)
                window_introns = get_window_intron_counts (bed_filename, mm_filename, rr_filename, flank_len, weighted_average)
                with open (out_filename, 'w') as out_file:
                    json.dump (window_introns, out_file)

def download_file_if_absent (url, dirname):
    filename = dirname + '/' + re.compile('^.*/').sub('',url)
    if os.path.isfile(filename):
        logging.warning(f"{filename} already downloaded")
    else:
        logging.warning(f"Downloading {url} to {dirname}/")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filename, 'wb') as out_file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    out_file.write(data)
    return filename

def get_window_intron_counts (bed_filename, mm_filename, rr_filename, flank_len, weighted_average):
    # Read windows
    logging.warning(f"Reading windows from {bed_filename}")
    windows_bedtool = pybedtools.BedTool(bed_filename)

    # Assign unique index to each window
    n_windows = 0
    def assign_window_idx(feature):
        nonlocal n_windows
        feature.append (str(n_windows))
        feature.start -= flank_len
        feature.end += flank_len
        n_windows += 1
        return feature
    windows_bedtool = windows_bedtool.each(assign_window_idx)

    # Read intron coords
    logging.warning(f"Reading intron coords from {rr_filename}")
    with gzip.open(rr_filename,"rt") as rr_file:
        rr_header, *rr_lines = rr_file.readlines()
    rr_col = dict((k,i) for (i,k) in enumerate(rr_header.split('\t')))
    rr_chrom_col, rr_start_col, rr_end_col, rr_strand_col = [rr_col[k] for k in ['chromosome', 'start', 'end', 'strand']]
    rr_lines = [l.split('\t') for l in rr_lines]
    
    rr_bed_list = [(l[rr_chrom_col], str(int(l[rr_start_col]) - 1), l[rr_end_col], str(n)) for n, l in enumerate(rr_lines)]
    rr_bedtool = pybedtools.BedTool(rr_bed_list)
    
    n_junctions = len(rr_bedtool)

    if weighted_average:
        # Scan through junction counts matrix once to get total counts for each sample
        # (We can't do this in-memory as the matrix is too big)
        logging.warning(f"Reading total sample counts from {mm_filename}")
    else:
        # If we're not up-weighting sparse samples, we don't need to parse the whole file; just the header
        logging.warning(f"Reading header from {mm_filename}")
    with gzip.open(mm_filename,"rt") as mm_file:
        for line in mm_file:
            if line != "" and line[0] != '%':
                break
        n_rows, n_samples, total_count = [int(f) for f in line.split('\t')]
        if n_junctions != n_rows:
            logging.warning(f"Number of rows ({n_rows}) in {mm_filename} does not match number of junctions ({n_junctions}) in {rr_filename}")
        tqdm_args = {"total": n_junctions, "unit": "rows", "unit_scale": True}
        if weighted_average:
            total_counts = np.zeros ((n_samples,), dtype=float)
            last_junction = 0
            n_lines = 0
            with tqdm(**tqdm_args) as progress_bar:
                for line in mm_file:
                    n_lines = n_lines + 1
                    fields = line.split('\t')
                    if len(fields) == 3:
                        n_junction, n_sample, count = [int(f) for f in fields]
                        if n_junction > last_junction:
                            progress_bar.update (n_junction - last_junction)
                            last_junction = n_junction
                        total_counts[n_sample - 1] += count
                    else:
                        logging.warning("Could not parse line: " + line)
            epsilon = 1e-10  # guard against NaNs
            sample_weight = (np.mean(total_counts) / n_samples) / [c or epsilon for c in total_counts]
            tqdm_args = {"total": n_lines, "unit": "lines", "unit_scale": True}
        else:
            sample_weight = 1 / np.ones ((n_samples,), dtype=float)

    # Read junction counts matrix again, normalizing and averaging across samples
    junction_count = np.zeros ((n_junctions,), dtype=float)
    logging.warning(f"Reading intron-sample counts from {mm_filename}")
    with gzip.open(mm_filename,"rt") as mm_file:
        for line in mm_file:
            if line != "" and line[0] != '%':
                break
        last_junction = 0
        with tqdm(**tqdm_args) as progress_bar:
            for line in mm_file:
                if weighted_average:
                    progress_bar.update(1)
                fields = line.split('\t')
                if len(fields) == 3:
                    n_junction, n_sample, count = [int(f) for f in fields]
                    junction_count[n_junction - 1] += count * sample_weight[n_sample - 1]
                    if not weighted_average and n_junction > last_junction:
                        progress_bar.update (n_junction - last_junction)
                        last_junction = n_junction
                else:
                    logging.warning("Could not parse line: " + line)

    # Find containments
    logging.warning(f"Mapping introns to windows")
    windows_rr_intersect = windows_bedtool.intersect(rr_bedtool,F=1,loj=True)
    
    # For each window in the training set, create a list of contained (donor_offset,acceptor_offset,count) tuples
    # The donor_offset is the coordinate of the first base in the intron, and the acceptor_offset is the coordinate of the last
    # If the intron is on the reverse strand then donor_offset > acceptor_offset
    # All coordinates are zero-based and within the reference frame of the window
    window_introns = [[] for _ in range(n_windows)]
    for window in windows_rr_intersect:
        try:
            if window.fields[6] == '-1':   # Null feature?
                continue
            window_start, window_idx, junction_start, junction_end, junction_idx = [int(window.fields[n]) for n in (1,4,6,7,8)]
            if window_idx >= n_windows:
                logging.warning(f"uh oh... {window.fields}")
                breakpoint()
            junction_start -= window_start
            junction_end -= window_start + 1  # we want the coordinate of the last base, not the BED-style "last base plus one"
            junction_strand = rr_lines[junction_idx][rr_strand_col]
            (donor_pos,acceptor_pos) = (junction_start,junction_end) if junction_strand == '+' else (junction_end,junction_start)
            window_introns[window_idx].append ((donor_pos, acceptor_pos, junction_count[junction_idx]))
        except:
            logging.warning(f"Could not parse window/junction coordinates: {window.fields}")

    return window_introns

if __name__ == "__main__":
    CLI(main, parser_mode='jsonnet')