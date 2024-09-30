import os
import sys
import json
import logging
from itertools import chain
import numpy as np

sys.path.append (os.path.dirname(__file__) + '/..')
from state import augment_xxj_counts, pad_xxj_counts

def concatenate (list_of_lists):
    return list(chain(*list_of_lists))

if len(sys.argv) < 3:
    print ("Usage: python test_augment_xxj.py processed_intron_counts.json windows...")
    sys.exit(1)

filename = sys.argv[1]
window_ids = sys.argv[2:]
print("Window IDs:", window_ids)

logging.warning("Loading %s", filename)
with open(filename, 'r') as f:
    intron_counts = json.load(f)

batch = [intron_counts['windows'][id] for id in window_ids]

n_out_bins = 8192
n_out_tracks = max (len(b[2]) for b in batch)

xxj_coords = np.array (concatenate ([concatenate ([[[n_batch, n_track, track[0][dac[0]], track[1][dac[1]]] for dac in track[2]] for n_track, track in enumerate(xxj_data)]) for n_batch, xxj_data in enumerate(batch)]), dtype=np.uint16)
xxj_counts = np.array (concatenate ([concatenate ([[dac[2] for dac in track[2]] for track in xxj_data]) for n_batch, xxj_data in enumerate(batch)]), dtype=np.float16)

print("Before augmentation:")
print(xxj_coords)
print(xxj_counts)

xxj_coords, xxj_counts = augment_xxj_counts (xxj_coords, xxj_counts, n_batches=len(batch), n_out_bins=n_out_bins, n_out_tracks=n_out_tracks)
xxj_coords, xxj_counts_weights = pad_xxj_counts (xxj_coords, xxj_counts)

print("After augmentation:")
print(xxj_coords)
print(xxj_counts_weights)
