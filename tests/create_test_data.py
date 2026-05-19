"""One-time script to create cropped test data from sample recording.

Run with: conda run -n mne python tests/create_test_data.py
"""
import os
import shutil
import mne
import numpy as np
from scipy.io import savemat

# Source data
src_dir = r'C:\Users\hasla\Desktop\old\björn_04_06_25\P01\data'
src_vhdr = os.path.join(src_dir, 'task_stim.vhdr')

# Destination
dst_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(dst_dir, exist_ok=True)

# Load raw data
raw = mne.io.read_raw_brainvision(src_vhdr, preload=True)
events, event_id = mne.events_from_annotations(raw)
sfreq = raw.info['sfreq']

# We want enough complete trials. The trial pattern is:
# S15 -> S2/S3 -> S16 -> S17/S18 -> (next trial)
# Find the sample of the 20th S15 event and crop a bit after its S17/S18 outcome
s15_events = events[events[:, 2] == event_id['Stimulus/S 15']]
n_trials_wanted = 20

if len(s15_events) < n_trials_wanted:
    raise RuntimeError(f'Not enough S15 events: {len(s15_events)}')

# Find the end marker (S17 or S18) after the last wanted trial
last_trial_sample = s15_events[n_trials_wanted - 1, 0]
# Find events after the last trial start
subsequent = events[events[:, 0] > last_trial_sample]
# Find the first S17 or S18 after the last trial
s17_id = event_id['Stimulus/S 17']
s18_id = event_id['Stimulus/S 18']
outcome_events = subsequent[(subsequent[:, 2] == s17_id) | (subsequent[:, 2] == s18_id)]
if len(outcome_events) == 0:
    raise RuntimeError('No outcome event found after last trial')

end_sample = outcome_events[0, 0]
# Add 2 seconds buffer after the last outcome
tmax_crop = (end_sample / sfreq) + 2.0
tmin_crop = 0.0

raw_cropped = raw.copy().crop(tmin=tmin_crop, tmax=min(tmax_crop, raw.times[-1]))

# Export as BrainVision format
dst_vhdr = os.path.join(dst_dir, 'task_stim.vhdr')
raw_cropped.export(dst_vhdr, fmt='brainvision', overwrite=True)

# Copy calibration .mat files
for mat_file in ['P_TARGET_64.mat', 'flip.mat', 'exclude_idx.mat']:
    src_path = os.path.join(src_dir, mat_file)
    dst_path = os.path.join(dst_dir, mat_file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f'Copied {mat_file}')
    else:
        print(f'WARNING: {mat_file} not found in source directory')

# Verify the cropped data
raw_check = mne.io.read_raw_brainvision(dst_vhdr, preload=True)
events_check, eid_check = mne.events_from_annotations(raw_check)
n_s15 = np.sum(events_check[:, 2] == eid_check.get('Stimulus/S 15', -1))
n_s16 = np.sum(events_check[:, 2] == eid_check.get('Stimulus/S 16', -1))
duration = raw_check.times[-1]

print(f'\nTest data created successfully:')
print(f'  Duration: {duration:.1f} s')
print(f'  S15 events (trial starts): {n_s15}')
print(f'  S16 events (responses): {n_s16}')
print(f'  Total events: {len(events_check)}')
print(f'  Channels: {len(raw_check.ch_names)}')
print(f'  Output: {dst_vhdr}')
