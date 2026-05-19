"""
Fixed vs Variable-Length Epochs in clamnibs
===========================================

This example shows how to use clamnibs with both fixed-length and
variable-length epochs. Fixed-length epochs are the traditional approach
(each trial has the same duration), while variable-length epochs allow
trials to span from one event marker to another, accommodating designs
where trial duration varies (e.g., reaction-time tasks).

In both cases, marker_definition is required to map trigger codes to
target phases — this is how the toolbox knows which stimulation phase
was applied on each trial.

Requirements
------------
- A BrainVision dataset (.vhdr) with event markers
- Calibration files (P_TARGET_64.mat, flip.mat) in the data folder
- An EEG dataset recorded with CLAM-NIBS

Marker structure assumed in this example
----------------------------------------
- S2 / S3 : condition codes (mapped to target phases 0 and pi)
- S15     : trial onset (appears simultaneously with S2 or S3)
- S16     : response (variable latency after S15)
"""

import numpy as np
from clamnibs.base import RawCLAM, EpochsCLAM, EpochsCLAMVariable
from clamnibs.beamformer import get_target
from clamnibs.source import compute_single_trial_amplitude, compute_single_trial_psd

# %% -----------------------------------------------------------------------
# 1. Variable-length epochs
# --------------------------------------------------------------------------
# Replace this path with the path to your own BrainVision .vhdr file.
vhdr_path = r'path\to\your\data\task_stim.vhdr'

# Define which trigger codes correspond to which target phases.
# This is required for both fixed and variable-length workflows.
marker_definition = {
    2: 0.0,      # S2 -> 0 rad
    3: np.pi,    # S3 -> pi rad
}

# For variable-length epochs, tmin/tmax are not needed on RawCLAM.
# They only control fixed-length epoch windows (EpochsCLAM).
raw = RawCLAM(
    vhdr_path,
    l_freq_target=8.0,
    h_freq_target=14.0,
    n_chs=64,
    design='trial_wise',
    ecg_channels=['ecg'],
    misc_channels=['envelope', 'envelope_am', 'eda'],
    marker_definition=marker_definition,
)

# Filter into the target frequency band.
raw_filtered = raw.copy().filter(8.0, 14.0)

# Create variable-length epochs: each trial spans from S15 (onset) to
# S16 (response), so duration varies with reaction time.
# The marker_definition is still used: each epoch is assigned its target
# phase based on the condition code (S2 or S3) near the start marker.
epochs_variable = EpochsCLAMVariable(
    raw_filtered,
    end_codes=[16],        # S16 = response marker (end of trial)
    start_codes=[15],      # S15 = trial onset (start of trial)
    tmin=0,                # offset relative to start marker (seconds)
    tmax=0,                # offset relative to end marker (seconds)
)

print(f'Variable epochs: {len(epochs_variable)} trials')
print(f'Durations (s):   {[f"{d:.2f}" for d in epochs_variable.durations]}')

# Each epoch's event code reflects the condition from marker_definition,
# not the start_code. This allows target phase lookup per epoch.
print(f'Event codes:     {epochs_variable.events[:, 2]}')
target_phases = [marker_definition[c] for c in epochs_variable.events[:, 2]]
print(f'Target phases:   {target_phases}')

# Data is returned as a list of arrays (one per epoch), since each epoch
# has a different number of time points.
data = epochs_variable.get_data()
print(f'Type of data:    {type(data).__name__}')
print(f'First epoch:     {data[0].shape}')
print(f'Second epoch:    {data[1].shape}')

# Extract the beamformed target signal for variable-length epochs.
# Returns a list of 1-D arrays instead of a 2-D array.
target_variable = get_target(epochs_variable)
print(f'\nTarget type:     {type(target_variable).__name__}')
print(f'Lengths:         {[len(t) for t in target_variable[:5]]}...')

# Compute single-trial amplitude using variable-length epochs.
# Pass end_codes to trigger variable-length mode.
df_amp_var = compute_single_trial_amplitude(
    raw_filtered, measure='hilbert_amp', end_codes=[16]
)
print('\nVariable-epoch amplitude results:')
print(df_amp_var.head())

# Compute single-trial PSD with variable-length epochs (requires broadband).
raw_broadband = raw.copy().filter(1.0, 30.0)
df_psd_var = compute_single_trial_psd(raw_broadband, end_codes=[16])
print('\nVariable-epoch PSD results:')
print(df_psd_var.head())

# %% -----------------------------------------------------------------------
# 2. Indexing and slicing variable-length epochs
# --------------------------------------------------------------------------
# You can index EpochsCLAMVariable just like a list.

single = epochs_variable[0]          # single epoch -> (n_channels, n_timepoints)
subset = epochs_variable[2:5]        # slice -> list of arrays
by_list = epochs_variable[[0, 3, 7]] # fancy indexing -> list of arrays

print(f'\nSingle epoch shape: {single.shape}')
print(f'Subset length:      {len(subset)}')
print(f'Fancy index length: {len(by_list)}')

# %% -----------------------------------------------------------------------
# 3. Fixed-length epochs (traditional approach)
# --------------------------------------------------------------------------
# For fixed-length epochs, tmin/tmax must be set on RawCLAM.
# Each epoch spans tmin to tmax (0 to 3 s) relative to the trigger.

raw_fixed = RawCLAM(
    vhdr_path,
    l_freq_target=8.0,
    h_freq_target=14.0,
    tmin=0,
    tmax=3,
    n_chs=64,
    design='trial_wise',
    ecg_channels=['ecg'],
    misc_channels=['envelope', 'envelope_am', 'eda'],
    marker_definition=marker_definition,
)

raw_fixed_filtered = raw_fixed.copy().filter(8.0, 14.0)
epochs_fixed = EpochsCLAM(raw_fixed_filtered)

print(f'\nFixed epochs: {len(epochs_fixed)} trials')
print(f'Data shape:   {epochs_fixed.get_data().shape}')
# -> (n_epochs, n_channels, n_timepoints), all epochs have the same length

# Extract the beamformed target signal for fixed epochs.
target_fixed = get_target(epochs_fixed)
print(f'Target shape: {target_fixed.shape}')
# -> (n_epochs, n_timepoints)

# Compute single-trial amplitude using fixed epochs.
df_amp_fixed = compute_single_trial_amplitude(raw_fixed_filtered, measure='hilbert_amp')
print('\nFixed-epoch amplitude results:')
print(df_amp_fixed.head())

# Compute single-trial PSD using fixed epochs (requires broadband filter).
raw_fixed_broadband = raw_fixed.copy().filter(1.0, 30.0)
df_psd_fixed = compute_single_trial_psd(raw_fixed_broadband)
print('\nFixed-epoch PSD results:')
print(df_psd_fixed.head())

# %% -----------------------------------------------------------------------
# 4. When to use which approach
# --------------------------------------------------------------------------
# Use FIXED-LENGTH epochs (EpochsCLAM) when:
#   - All trials have the same structure and duration
#   - You want a standard (n_epochs x n_channels x n_timepoints) array
#   - You are doing frequency-domain analyses that assume uniform length
#
# Use VARIABLE-LENGTH epochs (EpochsCLAMVariable) when:
#   - Trial duration depends on participant behavior (e.g., reaction time)
#   - You want to analyze only the task-relevant portion of each trial
#   - You need to avoid including post-response or inter-trial data
#
# Both approaches work with the same analysis functions
# (compute_single_trial_amplitude, compute_single_trial_psd, get_target).
# For variable-length mode, pass end_codes to the compute functions.
