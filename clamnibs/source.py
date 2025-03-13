import mne
import numpy as np
from scipy import linalg
from mne.filter import next_fast_len
from scipy.signal import hilbert
from pycircstat.descriptive import mean as circmean
import matplotlib.pyplot as plt
import pandas as pd
from .stats import _wrap
import seaborn as sns
from scipy.stats import ttest_ind
import emd
from math import degrees
from scipy.stats import f_oneway
from mne.time_frequency import psd_array_welch
from .misc import _get_ixs_goods, _get_main_target_phase
from .base import RawCLAM, EpochsCLAM
from .beamformer import get_target

def compute_phase_tracking(raw, plot=False):
    
    """
    Compute the phase lag between CLAM-NIBS and targeted brain oscillations.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    plot : bool, optional
        Whether to plot the phase tracking results (default is False).

    Returns:
    --------
    float
        The mean differene between the targeted phase lag and the actual phase lag, 
        computed across all targeted phase lags.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the data is not filtered into the target frequency range.
        If the function is called on data without CLAM-NIBS.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_phase_tracking can only be applied to RawCLAM objects')

    sfreq = raw.info['sfreq']
    l_freq = raw.info['highpass']
    h_freq = raw.info['lowpass']
    l_freq_target = raw.l_freq_target
    h_freq_target = raw.h_freq_target
    marker_definition = raw.marker_definition
    participant = raw.participant

    if not (l_freq == l_freq_target and h_freq == h_freq_target):
        raise Exception(
            'Data must be filtered into the target frequency range to compute phase tracking')
    if not raw.is_stim:
        raise Exception(
            'Phase tracking can only be computed for data with CLAM-tACS')

    df_phase = pd.DataFrame({'Target Phase': [], 'Actual Phase': []})
    target_codes = list(marker_definition.keys())
    target_phases = list(marker_definition.values())
    target_labels = ['{:d}'.format(int(degrees(x))) for x in target_phases]
    n_targets = len(target_codes)
    epochs_hil = EpochsCLAM(raw)
    target_hil = get_target(epochs_hil)
    envelope_hil = epochs_hil.get_data(['envelope']).squeeze()
    for target_code, target_label in zip(target_codes, target_labels):
        mask_condition = epochs_hil.events[:, 2] == target_code
        phasediffs = _wrap(
            np.angle(
                target_hil[mask_condition]) -
            np.angle(
                envelope_hil[mask_condition])).flatten()
        df_new = pd.DataFrame(
            {'Target Phase': [target_label] * len(phasediffs), 'Actual Phase': phasediffs})
        df_phase = pd.concat([df_phase, df_new])
    mean_actual_phases = [_wrap(circmean(df_phase.groupby('Target Phase').get_group(
        target_labels[ix])['Actual Phase'].to_numpy())) for ix in range(n_targets)]
    if plot:
        n_cols = int(np.ceil(np.sqrt(n_targets)))
        n_rows = int(np.ceil(n_targets / n_cols))     
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(
            4 * n_rows, 4 * n_cols), subplot_kw={'projection': 'polar'})
        axs = axs.flatten()
        for ix in range(n_targets):
            axs[ix].axvline(
                mean_actual_phases[ix],
                c='b',
                label='Mean Actual Phase' if ix == 0 else None)
            axs[ix].axvline(
                target_phases[ix],
                c='g',
                label='Target Phase' if ix == 0 else None)
            axs[ix].hist(
                df_phase.groupby('Target Phase').get_group(
                    target_labels[ix])['Actual Phase'].to_numpy(),
                color='k',
                alpha=0.3)
            axs[ix].set_title(target_labels[ix])
            axs[ix].yaxis.grid(False)
            axs[ix].xaxis.grid(False)
            axs[ix].get_yaxis().set_visible(False)
        for ix in range(n_targets, len(axs)):
            fig.delaxes(axs[ix])
        plt.legend()
        plt.suptitle('Actual Phase')
        plt.tight_layout()
    phase_delay = circmean(
        [tp - ap for tp, ap in zip(target_phases, mean_actual_phases)])
    return phase_delay

def compute_single_trial_amplitude(raw, measure='hilbert_amp'):
    
    """
    Compute single-trial amplitude of target oscillation for each CLAM-NIBS target phase.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute amplitude modulation. It can be one of the following:
            - 'hilbert_amp': Amplitude estimation based on Hilbert transform (default).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed amplitude values and CLAM-NIBS target phase for each epoch.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method for computing amplitude is not supported.
        If the Raw object does not meet the requirements for the chosen method.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_single_trial_amplitude can only be applied to RawCLAM objects')

    sfreq = raw.info['sfreq']
    l_freq = raw.info['highpass']
    h_freq = raw.info['lowpass']
    l_freq_target = raw.l_freq_target
    h_freq_target = raw.h_freq_target
    marker_definition = raw.marker_definition
    participant = raw.participant
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]

    if measure not in ['hilbert_amp']:
        raise Exception(
            'Method to compute amplitude must be \'hilbert_amp\'')
    if not (l_freq == l_freq_target and h_freq ==
            h_freq_target) and measure == 'hilbert_amp':
        raise Exception(
            'Raw object must be filtered into the target frequency range for amplitude estimation based on Hilbert')
    if design == 'trial_wise':
        epochs = EpochsCLAM(raw)
        target_hil = get_target(epochs)
    else:
        target_hil = get_target(raw.copy().apply_hilbert())
        target_hil = target_hil[None, :, :]
    epoch_amps = []
    for epoch_hil in target_hil:
        if measure == 'hilbert_amp':
            amp = np.mean(np.abs(epoch_hil))
        epoch_amps.append(amp)
    if design == 'trial_wise':
        epoch_target_phases = [marker_definition.get(x) for x in epochs.events[:, 2]]
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_amps),
                              'session': [session] * len(epoch_amps),
                              'design': [design] * len(epoch_amps),
                              'target_phase': epoch_target_phases,
                              'measure': [measure] * len(epoch_amps),
                              'value': epoch_amps})
    return df_result

def compute_single_trial_psd(raw):
    
    """
    Compute single-trial power spectral density of target oscillation for each CLAM-NIBS target phase.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed power spectral density along with CLAM-NIBS target phase for each epoch.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the Raw object does not meet the requirements for the chosen method.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_single_trial_psd can only be applied to RawCLAM objects')

    sfreq = raw.info['sfreq']
    l_freq = raw.info['highpass']
    h_freq = raw.info['lowpass']
    l_freq_target = raw.l_freq_target
    h_freq_target = raw.h_freq_target
    marker_definition = raw.marker_definition
    participant = raw.participant
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]

    if (l_freq > 1 or h_freq < 30):
        raise Exception(
            'Raw object must have a passband of at least 1 - 30 Hz for power spectral density estimation')
    if design == 'trial_wise':
        epochs = EpochsCLAM(raw, apply_hil=False)
        target = get_target(epochs)
    else:
        target = get_target(raw)
        target = target[None, :, :]
    assert np.isrealobj(target)
    epoch_psds = []
    for epoch in target:
        psd, freqs = psd_array_welch(x=epoch, 
                                     sfreq=sfreq,
                                     fmin=1, 
                                     fmax=30, 
                                     n_fft=np.min([epoch.shape[-1], int(2*sfreq)]))
        epoch_psds.append(psd)
    if design == 'trial_wise':
        epoch_target_phases = [marker_definition.get(x) for x in epochs.events[:, 2]]
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_psds),
                              'session': [session] * len(epoch_psds),
                              'design': [design] * len(epoch_psds),
                              'target_phase': epoch_target_phases,
                              'measure': ['psd'] * len(epoch_psds),
                              'value': epoch_psds})
    df_result.attrs['freqs'] = freqs
    df_result.attrs['l_freq_target'] = l_freq_target
    df_result.attrs['h_freq_target'] = h_freq_target
    return df_result