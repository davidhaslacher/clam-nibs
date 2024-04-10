import mne
import numpy as np
from scipy import linalg
from mne.filter import next_fast_len
from scipy.signal import hilbert
from scipy.stats import binom_test
from pycircstat.descriptive import mean as circmean
import matplotlib.pyplot as plt
import pandas as pd
from .stats import _wrap
import seaborn as sns
from scipy.stats import ttest_ind
import emd
from math import degrees
from scipy.stats import f_oneway
from pingouin import circ_corrcl
from fooof import FOOOF
from mne.time_frequency import psd_array_welch
from .misc import _get_ixs_goods, _get_main_target_phase
from .base import RawCLAM, get_epochs
from .beamformer import get_target


# TODO:
# Brain - heart coupling


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
    epochs_hil = get_epochs(raw)
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
        max_subplots_per_row = 3
        n_rows = n_targets // max_subplots_per_row + \
            min(1, n_targets % max_subplots_per_row)
        n_cols = min(n_targets, max_subplots_per_row)
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
        plt.legend()
        plt.suptitle('Actual Phase')
    phase_delay = circmean(
        [tp - ap for tp, ap in zip(target_phases, mean_actual_phases)])
    return phase_delay


def _get_fooof_amplitude(
        data,
        sfreq,
        l_freq,
        h_freq,
        l_freq_target,
        h_freq_target):
    psd, freqs = psd_array_welch(data, sfreq, l_freq, h_freq, int(2.5 * sfreq))
    fm = FOOOF(verbose=False)
    fm.fit(freqs, psd)
    peak_params = fm.get_results().peak_params
    peak_params = peak_params[(peak_params[:, 0] > l_freq_target) & (
        peak_params[:, 0] < h_freq_target)]
    amp = np.max(peak_params[:, 1])
    return amp


def _get_hht_amplitude(
        data,
        sfreq,
        l_freq,
        h_freq,
        l_freq_target,
        h_freq_target):
    imf = emd.sift.mask_sift(data, max_imfs=4)
    IP, IF, IA = emd.spectra.frequency_transform(imf, sfreq, 'hilbert')
    freqs_carrier, hht = emd.spectra.hilberthuang(
        IF, IA, (l_freq, h_freq, int(2.5 * (h_freq - l_freq))))
    amp = hht[(freqs_carrier > l_freq_target) & (
        freqs_carrier < h_freq_target)].max()
    return amp


def compute_amplitude_modulation(raw, measure='hilbert_amp'):
    
    """
    Compute the phase-lag dependent modulation of target oscillation amplitude by CLAM-NIBS.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute amplitude modulation. It can be one of the following:
            - 'hht_amp': Amplitude estimation based on Hilbert-Huang Transform.
            - 'fooof_amp': Amplitude estimation based on FOOOF algorithm.
            - 'hilbert_amp': Amplitude estimation based on Hilbert transform (default).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed amplitude values for each epoch.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method for computing amplitude is not supported.
        If the Raw object does not meet the requirements for the chosen method.
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
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]

    if measure not in ['hht_amp', 'fooof_amp', 'hilbert_amp']:
        raise Exception(
            'Method to compute amplitude must be \'hht_amp\', \'fooof_amp\', or \'hilbert_amp\'')
    if (l_freq > 1 or h_freq < 30) and measure in ['hht_amp', 'fooof_amp']:
        raise Exception(
            'Raw object must have a passband of at least 1 - 30 Hz for amplitude estimation based on HHT or FOOOF')
    if not (l_freq == l_freq_target and h_freq ==
            h_freq_target) and measure == 'hilbert_amp':
        raise Exception(
            'Raw object must be filtered into the target frequency range for amplitude estimation based on Hilbert')
    if design == 'trial_wise':
        epochs = get_epochs(raw)
        target_hil = get_target(epochs)
    else:
        target_hil = get_target(raw.copy().apply_hilbert())
        target_hil = target_hil[None, :, :]
    epoch_amps = []
    for epoch_hil in target_hil:
        if measure == 'hht_amp':
            amp = _get_hht_amplitude(
                np.real(epoch_hil),
                sfreq,
                l_freq,
                h_freq,
                l_freq_target,
                h_freq_target)
        elif measure == 'fooof_amp':
            amp = _get_fooof_amplitude(
                np.real(epoch_hil),
                sfreq,
                l_freq,
                h_freq,
                l_freq_target,
                h_freq_target)
        else:
            amp = np.mean(np.abs(epoch_hil))
        epoch_amps.append(amp)
    if design == 'trial_wise':
        epoch_target_phases = np.vectorize(
            marker_definition.get)(epochs.events[:, 2])
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_amps),
                              'session': [session] * len(epoch_amps),
                              'design': [design] * len(epoch_amps),
                              'target_phase': epoch_target_phases,
                              'measure': [measure] * len(epoch_amps),
                              'value': epoch_amps})
    return df_result


def _get_fooof_exponent(
        data,
        sfreq,
        l_freq,
        h_freq,
        l_freq_target,
        h_freq_target):
    psd, freqs = psd_array_welch(data, sfreq, l_freq, h_freq, int(2.5 * sfreq))
    fm = FOOOF(verbose=False)
    fm.fit(freqs, psd)
    ap = fm.get_results().aperiodic_params[-1]
    return ap


def compute_aperiodic_modulation(raw, measure='fooof_ae'):
    
    """
    Compute the phase-lag dependent modulation of target aperiodic activity by CLAM-NIBS.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute the aperiodic exponent. It can only be 'fooof_ae' (default).
    plot : bool, optional
        Whether to plot the aperiodic modulation results (default is False).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed aperiodic exponent for each epoch.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method for computing the aperiodic exponent is not supported.
        If the Raw object does not meet the requirements for the chosen method.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_aperiodic_modulation can only be applied to RawCLAM objects')

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

    if measure not in ['fooof_ae']:
        raise Exception(
            'Method to compute aperiodic exponent must be \'fooof_ae\'')
    if (l_freq > 1 or h_freq < 30) and measure in ['fooof_ae']:
        raise Exception(
            'Raw object must have a passband of at least 1 - 30 Hz for aperiodic exponent estimation based on FOOOF')
    if design == 'trial_wise':
        epochs = get_epochs(raw)
        target_hil = get_target(epochs)
    else:
        target_hil = get_target(raw.copy().apply_hilbert())
        target_hil = target_hil[None, :, :]
    epoch_aes = []
    for epoch_hil in target_hil:
        ap = _get_fooof_exponent(
            np.real(epoch_hil),
            sfreq,
            l_freq,
            h_freq,
            l_freq_target,
            h_freq_target)
        epoch_aes.append(ap)
    if design == 'trial_wise':
        epoch_target_phases = np.vectorize(
            marker_definition.get)(epochs.events[:, 2])
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_aes),
                              'session': [session] * len(epoch_aes),
                              'design': [design] * len(epoch_aes),
                              'target_phase': epoch_target_phases,
                              'measure': [measure] * len(epoch_aes),
                              'value': epoch_aes})
    return df_result


def _get_fooof_frequency(
        data,
        sfreq,
        l_freq,
        h_freq,
        l_freq_target,
        h_freq_target):
    psd, freqs = psd_array_welch(data, sfreq, l_freq, h_freq, int(2.5 * sfreq))
    fm = FOOOF(verbose=False)
    fm.fit(freqs, psd)
    peak_params = fm.get_results().peak_params
    peak_params = peak_params[(peak_params[:, 0] > l_freq_target) & (
        peak_params[:, 0] < h_freq_target)]
    pf = peak_params[np.argmax(peak_params[:, 1])][0]
    return pf


def compute_frequency_modulation(raw, measure='fooof_pf'):
    
    """
    Compute the phase-lag dependent modulation of target oscillation frequency by CLAM-NIBS.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute the peak frequency. It can only be 'fooof_pf' (default).
    plot : bool, optional
        Whether to plot the frequency modulation results (default is False).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed peak frequency values for each epoch.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method for computing the peak frequency is not supported.
        If the Raw object does not meet the requirements for the chosen method.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_frequency_modulation can only be applied to RawCLAM objects')

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

    if measure not in ['fooof_pf']:
        raise Exception(
            'Method to compute peak frequency must be \'fooof_pf\'')
    if (l_freq > 1 or h_freq < 30) and measure in ['fooof_pf']:
        raise Exception(
            'Raw object must have a passband of at least 1 - 30 Hz for peak frequency estimation based on FOOOF')
    if design == 'trial_wise':
        epochs = get_epochs(raw)
        target_hil = get_target(epochs)
    else:
        target_hil = get_target(raw.copy().apply_hilbert())
        target_hil = target_hil[None, :, :]
    epoch_pfs = []
    for epoch_hil in target_hil:
        ap = _get_fooof_frequency(
            np.real(epoch_hil),
            sfreq,
            l_freq,
            h_freq,
            l_freq_target,
            h_freq_target)
        epoch_pfs.append(ap)
    if design == 'trial_wise':
        epoch_target_phases = np.vectorize(
            marker_definition.get)(epochs.events[:, 2])
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_pfs),
                              'session': [session] * len(epoch_pfs),
                              'design': [design] * len(epoch_pfs),
                              'target_phase': epoch_target_phases,
                              'measure': [measure] * len(epoch_pfs),
                              'value': epoch_pfs})
    return df_result