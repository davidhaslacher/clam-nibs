import numpy as np
from scipy import linalg
import mne
from scipy.signal import hilbert
from . import misc
from mne.stats import permutation_cluster_test
from scipy.sparse import coo_matrix
import scipy
from mne.stats import ttest_ind_no_p
from pingouin import circ_corrcl
from mne.viz.topomap import _get_pos_outlines
from mne.utils.check import _check_sphere
import matplotlib.pyplot as plt
import seaborn as sns
from mne.viz import plot_sensors
import pandas as pd
from .misc import _get_ixs_goods, _get_main_target_phase
from .base import RawCLAM, EpochsCLAM
from .source import get_target
from tqdm import tqdm


def _find_n_nulls(A, B, D, M):
    errors = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        errors.append(linalg.norm(B - P @ A @ P.T))
    return np.argmin(errors)


def clean_sensor_data(obj_no_stim, obj_stim):
    
    """
    Use Stimulation Artifact Source Separation (SASS) [1] to remove electric stimulation artifacts 
    from EEG sensor data recorded during CLAM-NIBS. Bad sensors are excluded from the procedure,
    and interpolated after application of SASS.

    Parameters:
    -----------
    obj_no_stim : RawCLAM or EpochsCLAM
        The RawCLAM or EpochsCLAM object containing the sensor data without CLAM-NIBS stimulation.
    obj_stim : RawCLAM or EpochsCLAM
        The RawCLAM or EpochsCLAM object containing the sensor data with CLAM-NIBS stimulation.

    Raises:
    -------
    Exception
        If the input data types are not RawCLAM or EpochsCLAM objects.
        If the data with and without CLAM-NIBS have different bandpass filter or target frequency range settings.
    
    References:
    -----------
    [1] Haslacher, David, et al. "Stimulation artifact source separation (SASS) for assessing electric brain oscillations 
    during transcranial alternating current stimulation (tACS)." Neuroimage 228 (2021): 117571.
    """
    
    if not (isinstance(obj_no_stim, RawCLAM) or isinstance(obj_no_stim, EpochsCLAM)) and \
            (isinstance(obj_stim, RawCLAM) or isinstance(obj_stim, EpochsCLAM)):
        raise Exception('clean_sensor_data can only be applied to RawCLAM or EpochsCLAM objects')

    equal_l_freq = obj_no_stim.info['highpass'] == obj_stim.info['highpass']
    equal_h_freq = obj_no_stim.info['lowpass'] == obj_stim.info['lowpass']
    equal_l_freq_target = obj_no_stim.l_freq_target == obj_stim.l_freq_target
    equal_h_freq_target = obj_no_stim.h_freq_target == obj_stim.h_freq_target
    if not (
            equal_l_freq and equal_h_freq and equal_l_freq_target and equal_h_freq_target):
        raise Exception(
            'Data with and without CLAM-tACS have different bandpass-filter or target frequency range settings')
    l_freq = obj_no_stim.info['highpass']
    h_freq = obj_no_stim.info['lowpass']
    l_freq_target = obj_no_stim.l_freq_target
    h_freq_target = obj_no_stim.h_freq_target
    ixs_goods = _get_ixs_goods(obj_no_stim)

    if not (l_freq == l_freq_target and h_freq == h_freq_target):
        raise Exception(
            'Data must be filtered into the target frequency range to apply SASS')
        
    if isinstance(obj_stim, RawCLAM):
        A = np.cov(obj_stim.get_data(ixs_goods))
    else:
        A = np.cov(
            np.concatenate(
                obj_stim.get_data(ixs_goods),
                axis=-1))
        
    if isinstance(obj_no_stim, RawCLAM):
        B = np.cov(obj_no_stim.get_data(ixs_goods))
    else:
        B = np.cov(
            np.concatenate(
                obj_no_stim.get_data(ixs_goods),
                axis=-1))
    
    eigen_values, eigen_vectors = linalg.eig(A, B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:, ix].T
    M = linalg.pinv(D)
    n_nulls = _find_n_nulls(A, B, D, M)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)

    if isinstance(obj_stim, RawCLAM):
        obj_stim._data[ixs_goods] = P @ obj_stim._data[ixs_goods]
    else:
        obj_stim._data[:, ixs_goods] = np.array(
            [P @ epoch for epoch in obj_stim._data[:, ixs_goods]])
    
    obj_stim.interpolate_bads(reset_bads=True)


def compute_single_trial_connectivity(raw, measure='phase_lag_index'):
    
    """
    Compute single-trial amplitude of target oscillation and assign it to CLAM-NIBS target phase.

    Parameters:
    -----------
    raw : RawCLAM
        The RawCLAM object containing the raw data to analyze.
    measure : str, optional
        The connectivity measure to compute. Currently, only 'phase_lag_index' is supported (default).

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

    if measure not in ['phase_lag_index']:
        raise Exception('Connectivity measure must be \"phase_lag_index\"')
    if not (l_freq == l_freq_target and h_freq == h_freq_target):
        raise Exception(
            'Data must be filtered into the target frequency range to compute connectivity')
    if not len(raw.info['bads']) == 0:
        raise Exception(
            'Bads must be interpolated before connectivity computation')
        
    if design == 'trial_wise':
        epochs = EpochsCLAM(raw)
        data_hil = epochs.get_data(picks='eeg')
    else:
        data_hil = raw.copy().apply_hilbert()[None, :, :]
        
    phases = np.angle(data_hil)
    n_chs = raw.n_chs
    
    conns = []
    for phase in tqdm(phases, desc='Computing single-trial connectivity'):
        conn = np.zeros((n_chs, n_chs))
        for ix1 in range(n_chs):
            for ix2 in range(ix1 + 1, n_chs):
                conn[ix1, ix2] = misc._pli(phase[ix1], phase[ix2])
        conn += conn.T
        conn += np.diag(np.ones(n_chs))
        conns.append(conn)
        
    if design == 'trial_wise':
        events = epochs.events
        target_phases = np.vectorize(marker_definition.get)(events[:, 2])
    else:
        events = mne.events_from_annotations(raw)[0]
        target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(conns),
                              'session': [session] * len(conns),
                              'design': [design] * len(conns),
                              'target_phase': target_phases,
                              'measure': [measure] * len(conns),
                              'value': conns})
    return df_result