from mne.preprocessing import find_ecg_events
import mne
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import degrees
from statistics import mode
from .base import RawCLAM, EpochsCLAM
from scipy.io import savemat
import warnings
from scipy.stats import zscore

def concat_dfs(dfs):
    attrs = dfs[-1].attrs
    for df in dfs:
        df.attrs = {}
    df = pd.concat(dfs)
    df.attrs = attrs
    return df

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

def _fmt(string):
    return string.replace(
        '_',
        ' ').title().replace(
        ' Of ',
        ' of ').replace(
            ' And ',
        ' and ')

def _get_trial_target_codes_cwm_error(raw):
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

def _get_trial_target_codes_binary_accuracy(raw, correct_codes, incorrect_codes):
    events = mne.events_from_annotations(raw)[0]
    n_events = len(events)
    target_codes = raw.marker_definition.keys()
    trial_target_codes = []
    trial_accuracies = []
    for ix_event in range(n_events):
        event = events[ix_event]
        if event[2] in target_codes:
            for ix_event_post in range(ix_event+1, n_events):
                event_post = events[ix_event_post]
                if event_post[2] in correct_codes:
                    trial_target_codes.append(event[2])
                    trial_accuracies.append(1)
                    break
                if event_post[2] in incorrect_codes:
                    trial_target_codes.append(event[2])
                    trial_accuracies.append(0)
                    break
                if event_post[2] in target_codes:
                    warnings.warn("A target event code marking a trial was found \
                                  without a following correct/incorrect marker", UserWarning)
                    break
    trial_target_codes, trial_accuracies = np.array(
        trial_target_codes), np.array(trial_accuracies)
    return trial_target_codes, trial_accuracies

def compute_single_trial_behavior(raw, measure='binary_accuracy', correct_codes=[10], incorrect_codes=[11, 12]):
    
    """
    Compute single-trial task performance and assign it to CLAM-NIBS target phase.
    Applicable to any task with trial-by-trial responses.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The method used to compute task performance. Either 'binary' (default) or 'cwm'.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed performance values and CLAM-NIBS target phase for each trial.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the modulation of task performance is attempted on data without CLAM-tACS stimulation.
        
    Notes:
    ------
    - This function only works if the data contains a target phase marker at the beginning of each trial.
    """
    
    if measure not in ['binary_accuracy', 'cwm_error']:
        raise Exception(
            'Method to compute working memory error must be \'binary_accuracy\' or \'cwm_error\'')
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_single_trial_behavior can only be applied to RawCLAM objects')
    marker_definition = raw.marker_definition
    participant = raw.participant
    session = raw.session
    design = raw.design
    if not raw.is_stim:
        raise Exception(
            'Single-trial performance with target phases can only be computed on data with CLAM-NIBS')
    match measure:
        case 'binary_accuracy':
            trial_target_codes, trial_values = _get_trial_target_codes_binary_accuracy(raw, correct_codes, incorrect_codes)
        case 'cwm_error':
            trial_target_codes, trial_values = _get_trial_target_codes_cwm_error(raw)
    trial_values = np.abs(trial_values)
    trial_target_phases = [marker_definition.get(x) for x in trial_target_codes]
    if design == 'session_wise':
        main_trial_code = mode(trial_target_codes)
        mask = trial_target_codes == main_trial_code
        trial_target_codes = trial_target_codes[mask]
        trial_values = trial_values[mask]
    df_result = pd.DataFrame({'participant': [participant] * len(trial_values),
                              'session': [session] * len(trial_values),
                              'design': [design] * len(trial_values),
                              'target_phase': trial_target_phases,
                              'measure': [measure] * len(trial_values),
                              'value': trial_values})
    return df_result

def compute_single_trial_rr(raw):
    
    """
    Compute single-trial RR-intervals and assign them to CLAM-NIBS target phase.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed RR intervals and CLAM-NIBS target phases.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method is attempted on data without CLAM-tACS stimulation.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_single_trial_rr can only be applied to RawCLAM objects')
    marker_definition = raw.marker_definition
    target_codes = list(marker_definition.keys())
    sfreq = raw.info['sfreq']
    tmin = raw.tmin
    tmax = raw.tmax
    participant = raw.participant
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]
    
    if not raw.is_stim:
        raise Exception(
            'Single-trial RR intervals can only be computed on data with CLAM-tACS')
        
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
                trial_target_codes.extend([ev_trial[2]] * len(this_trial_rrs))
                trial_rrs.extend(this_trial_rrs)
        trial_target_phases = [marker_definition.get(x) for x in trial_target_codes]
    else:
        events_ecg = find_ecg_events(raw, ch_name='ecg')[0]
        trial_rrs = np.diff(events_ecg[:,0]) / sfreq
        main_target_phase = _get_main_target_phase(marker_definition, events)
        trial_target_phases = [main_target_phase] * len(trial_rrs)
    trial_rrs = np.array(trial_rrs)
    trial_target_phases = np.array(trial_target_phases, dtype=object)
    zscores = zscore(trial_rrs)
    mask = (zscores > -1.6) & (zscores < 1.6)
    trial_rrs = trial_rrs[mask]
    trial_target_phases = trial_target_phases[mask]
    df_result = pd.DataFrame({'participant': [participant] * len(trial_rrs),
                              'session': [session] * len(trial_rrs),
                              'design': [design] * len(trial_rrs),
                              'target_phase': trial_target_phases,
                              'measure': ['rr_interval'] * len(trial_rrs),
                              'value': trial_rrs})
    return df_result

def compute_single_trial_scr(raw):
    
    """
    Compute single-trial area under the curve (AUC) of the skin conductance response (SCR) [1],
    and assign it to CLAM-NIBS target phase.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed SCR AUC and CLAM-NIBS target phases.

    Raises:
    -------
    Exception
        If the input raw data is not of type RawCLAM.
        If the method is attempted on data without CLAM-tACS stimulation.
        
    [1] Bach, Dominik R., Karl J. Friston, and Raymond J. Dolan. "Analytic measures for quantification of arousal 
        from spontaneous skin conductance fluctuations." International journal of psychophysiology 76.1 (2010): 52-55.
    
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('compute_single_trial_scr can only be applied to RawCLAM objects')
    marker_definition = raw.marker_definition
    target_codes = list(marker_definition.keys())
    sfreq = raw.info['sfreq']
    tmin = raw.tmin
    tmax = raw.tmax
    participant = raw.participant
    session = raw.session
    design = raw.design
    events = mne.events_from_annotations(raw)[0]
    
    if not raw.is_stim:
        raise Exception(
            'Single-trial SCR AUC can only be computed on data with CLAM-tACS')
        
    raw = raw.copy()
    raw.pick_channels(['eda'])
    raw.filter(0.1, 5, picks='all')
    raw._data -= raw._data.min()
    
    if design == 'trial_wise':
        epochs = EpochsCLAM(raw, apply_hil = False)
        epoch_aucs = epochs.get_data().squeeze().sum(-1)
    else:
        epoch_aucs = raw.get_data().squeeze().sum(-1)
        
    if design == 'trial_wise':
        epoch_target_phases = [marker_definition.get(x) for x in epochs.events[:, 2]]
    else:
        epoch_target_phases = [_get_main_target_phase(marker_definition, events)]
    
    df_result = pd.DataFrame({'participant': [participant] * len(epoch_aucs),
                            'session': [session] * len(epoch_aucs),
                            'design': [design] * len(epoch_aucs),
                            'target_phase': epoch_target_phases,
                            'measure': ['scr_auc'] * len(epoch_aucs),
                            'value': epoch_aucs})
    
    return df_result
    
def save_calibration_data(obj, folder_path, phase_delay=None):
    exclude_idx_file_path = '{}\\exclude_idx.mat'.format(folder_path)
    p_target_file_path = '{}\\P_TARGET_{:d}.mat'.format(folder_path, int(obj.n_chs))
    flip_file_path = '{}\\flip.mat'.format(folder_path)
    
    
    exclude_idx = np.sort([obj.ch_names.index(ch) + 1 for ch in obj.info["bads"]])
    data_dict = {"exclude_idx": exclude_idx}
    savemat(exclude_idx_file_path, data_dict)
    
    data_dict = {'P_TARGET_%i'%obj.n_chs:obj.forward_full}
    savemat(p_target_file_path, data_dict)
    
    data_dict = {'flip':obj.flip}
    savemat(flip_file_path, data_dict)
    
    if phase_delay is not None:
        phase_delay_file_path = '{}\\phase_delay.mat'.format(folder_path)
        
        data_dict = {'phase_delay':phase_delay}
        savemat(phase_delay_file_path, data_dict)
        

def df_to_array(df_data):
    """
    Convert a DataFrame containing one data point (trial or participant) per row to an ndarray.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The DataFrame object containing the data to convert.

    Returns:
    --------
    numpy.ndarray
        An ndarray of shape (n_phases, n_datapoints, ...).
    """
    
    gb_target_phases = df_data.sort_values(
        'target_phase').groupby('target_phase')
    target_phases = []
    data = []
    for target_phase, df_target_phase in gb_target_phases:
        target_phases.append(target_phase)
        data.append(np.stack(df_target_phase['value']))
    n_epochs = np.min([d.shape[0] for d in data])
    data = np.array([d[:n_epochs] for d in data])
    # data is now (n_phases, n_epochs, ...)
    return data, target_phases