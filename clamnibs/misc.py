from mne.preprocessing import find_ecg_events
import mne
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import degrees
from pingouin import circ_corrcl
from statistics import mode
from .base import RawCLAM

def _get_main_target_phase(marker_definition, events):
    target_codes = marker_definition.keys()
    events = events[np.isin(events[:, 2], target_codes)]
    trial_codes = events[:-1, 2]
    trial_durations = np.diff(events[:, 0])
    total_durations = []
    for target_code in target_codes:
        total_durations.append(trial_durations[trial_codes == target_code].sum())
    main_code = target_codes[np.argmax(total_durations)]
    return marker_definition[main_code]


def _get_ixs_goods(obj):
    types = obj.get_channel_types()
    if not np.all(np.array(types[:obj.n_chs]) == 'eeg'):
        raise Exception('Data should contain all EEG channels, including bads')
    return [ix for ix, ch in enumerate(
        obj.ch_names[:obj.n_chs]) if ch not in obj.info['bads']]


def _wrap(phase):
    return np.angle(np.exp(1j * phase))


def _circmean(phase):
    return np.angle(np.mean(np.exp(1j * phase)))


def _pli(phase1, phase2):
    return np.mean(np.sign(_wrap(phase1 - phase2)))


def _get_trial_target_codes_errors(raw):
    events = mne.events_from_annotations(raw)[0]
    n_events = len(events)
    target_codes = raw.marker_definition.keys()
    trial_target_codes = []
    trial_errors = []
    for ix_event in range(n_events):
        if ix_event < n_events - 3:
            event = events[ix_event]
            if event[2] in target_codes:
                # assert np.all(events[ix_event+1:ix_event+13,2]>=50)
                true_color = np.sum(events[ix_event + 1:ix_event + 7, 2]) - 300
                # assert -1 <= true_color <= 366
                est_color = np.sum(events[ix_event + 7:ix_event + 13, 2]) - 300
                # assert -1 <= est_color <= 366
                error = true_color - est_color
                if error < -180:
                    error += 360
                if error > 180:
                    error -= 360
                assert -180 <= error <= 180
                trial_target_codes.append(event[2])
                trial_errors.append(error)
    trial_target_codes, trial_errors = np.array(
        trial_target_codes), np.array(trial_errors)
    return trial_target_codes, trial_errors

def compute_wm_error_modulation(raw, measure='cwm_error'):
    
    """
    Compute the phase-lag dependent modulation of behavioral error by CLAM-NIBS 
    in the continous working memory (WM) task.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute working memory error. It can only be 'cwm_error' (default).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed WM error values for each trial.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the modulation of WM error is attempted on data without CLAM-tACS stimulation.
        
    Notes:
    ------
    - This function only works if the data contains a target phase marker at the beginning of each trial.
    """
    
    if measure not in ['cwm_error']:
        raise Exception(
            'Method to compute working memory error must be \'cwm_error\'')
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_wm_error_modulation can only be applied to RawCLAM objects')
    marker_definition = raw.marker_definition
    participant = raw.participant
    session = raw.session
    design = raw.design
    if not raw.is_stim:
        raise Exception(
            'Modulation of WM error can only be computed on data with CLAM-tACS')
    target_codes = list(marker_definition.keys())
    target_phases = list(marker_definition.values())
    target_labels = ['{:d}'.format(int(degrees(x))) for x in target_phases]
    n_targets = len(target_codes)
    trial_target_codes, trial_errors = _get_trial_target_codes_errors(raw)
    trial_errors = np.abs(trial_errors)
    trial_target_phases = np.vectorize(
        marker_definition.get)(trial_target_codes)
    if design == 'session_wise':
        main_trial_code = mode(trial_target_codes)
        mask = trial_target_codes == main_trial_code
        trial_target_codes = trial_target_codes[mask]
        trial_errors = trial_errors[mask]
    df_result = pd.DataFrame({'participant': [participant] * len(trial_errors),
                              'session': [session] * len(trial_errors),
                              'design': [design] * len(trial_errors),
                              'target_phase': trial_target_phases,
                              'measure': [measure] * len(trial_errors),
                              'value': trial_errors})
    return df_result

def compute_rr_modulation(raw):
    
    """
    Compute the phase-lag dependent modulation of heart rate (RR-intervals) by CLAM-NIBS.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed RR interval modulation values for each trial.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method is attempted on data without CLAM-tACS stimulation.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_rr_modulation can only be applied to RawCLAM objects')
    marker_definition = raw.marker_definition
    target_codes = list(marker_definition.keys())
    target_phases = list(marker_definition.values())
    sfreq = raw.info['sfreq']
    tmin = raw.tmin
    tmax = raw.tmax
    participant = raw.participant
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]
    
    if not raw.is_stim:
        raise Exception(
            'Modulation of RR intervals can only be computed on data with CLAM-tACS')
        
    if design == 'trial_wise':
        n_targets = len(target_codes)
        events_ecg = find_ecg_events(raw, ch_name='ecg')[0]
        events_trials = mne.events_from_annotations(raw)[0]
        events_trials = events_trials[np.isin(events_trials[:, 2], target_codes)]
        trial_target_codes = []
        trial_rrs = []
        for ev_trial in events_trials:
            this_trial_rpeaks = []
            for ev_ecg in events_ecg:
                if ev_ecg[0] > ev_trial[0] + tmin * \
                        sfreq and ev_ecg[0] < ev_trial[0] + tmax * sfreq:
                    this_trial_rpeaks.append(ev_ecg[0])
            if len(this_trial_rpeaks) >= 2:
                this_trial_rrs = np.diff(this_trial_rpeaks) / sfreq
                trial_target_codes.extend([ev_trial[:, 2]] * len(this_trial_rrs))
                trial_rrs.extend(this_trial_rrs)
        trial_target_phases = np.vectorize(
            marker_definition.get)(trial_target_codes)
    else:
        events_ecg = find_ecg_events(raw, ch_name='ecg')[0]
        trial_rrs = np.diff(events_ecg[:,0]) / sfreq
        main_target_phase = _get_main_target_phase(marker_definition, events)
        trial_target_phases = [main_target_phase] * len(trial_rrs)
    
    df_result = pd.DataFrame({'participant': [participant] * len(trial_rrs),
                              'session': [session] * len(trial_rrs),
                              'design': [design] * len(trial_rrs),
                              'target_phase': trial_target_phases,
                              'measure': ['rr_interval'] * len(trial_rrs),
                              'value': trial_rrs})
    return df_result