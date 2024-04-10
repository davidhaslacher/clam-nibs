import numpy as np
from scipy import linalg
import mne
from scipy.signal import hilbert
from misc import pli
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
    if isinstance(obj_stim, mne.Raw):
        A = np.cov(obj_stim.get_data([ixs_goods]))
    else:
        A = np.cov(
            np.concatenate(
                obj_stim.get_data([ixs_goods]),
                axis=-1))
    if isinstance(obj_no_stim, mne.Raw):
        B = np.cov(obj_no_stim.get_data([ixs_goods]))
    else:
        B = np.cov(
            np.concatenate(
                obj_no_stim.get_data([ixs_goods]),
                axis=-1))
    eigen_values, eigen_vectors = linalg.eig(A, B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:, ix].T
    M = linalg.pinv2(D)
    n_nulls = _find_n_nulls(A, B, D, M)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)

    if isinstance(obj_stim, mne.Raw):
        obj_stim._data[ixs_goods] = P @ obj_stim._data[ixs_goods]
    else:
        obj_stim._data[:, ixs_goods] = np.array(
            [P @ epoch for epoch in obj_stim._data[:, ixs_goods]])
    obj_stim.interpolate_bads(reset_bads=True)


def compute_connectivity(obj, measure='phase_lag_index'):
    
    """
    Compute connectivity between EEG sensors using the phase lag index [1].

    Parameters:
    -----------
    obj : RawCLAM or EpochsCLAM
        The RawCLAM or EpochsCLAM object containing the EEG data.
    measure : str, optional
        The connectivity measure to compute. Currently, only 'phase_lag_index' is supported (default).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the computed connectivity matrices for each trial or epoch.

    Raises:
    -------
    Exception
        If the input data type is not RawCLAM or EpochsCLAM.
        If the connectivity measure is not supported.
        If the data is not filtered into the target frequency range.
        If the data does not contain analytic signals for connectivity computation.
        If there are bad channels that have not been interpolated.
    
    References:
    -----------
    [1] Stam, Cornelis J., Guido Nolte, and Andreas Daffertshofer. "Phase lag index: assessment of 
    functional connectivity from multi channel EEG and MEG with diminished bias from common sources." 
    Human brain mapping 28.11 (2007): 1178-1193.
    """
    
    if not (isinstance(obj, RawCLAM) or isinstance(obj, EpochsCLAM)):
        raise Exception('compute_connectivity can only be applied to RawCLAM or EpochsCLAM objects')

    l_freq = obj.info['highpass']
    h_freq = obj.info['lowpass']
    l_freq_target = obj.l_freq_target
    h_freq_target = obj.h_freq_target
    marker_definition = obj.marker_definition
    participant = obj.participant
    session = obj.session
    design = obj.design
    events = obj.events
    bads = obj.info['bads']
    data = obj.get_data()

    if measure not in ['phase_lag_index']:
        raise Exception('Connectivity measure must be \"phase_lag_index\"')
    if not (l_freq == l_freq_target and h_freq == h_freq_target):
        raise Exception(
            'Data must be filtered into the target frequency range to compute connectivity')
    if not np.iscomplexobj(data):
        raise Exception(
            'Data must contain analytic signal for connectivity computation')
    if not len(bads) == 0:
        raise Exception(
            'Bads must be interpolated before connectivity computation')

    if data.ndim == 2:
        data = data[None, :, :]
    phases = np.angle(data)
    n_chs = phases.shape[1]
    conns = []
    for phase in phases:
        conn = np.zeros((n_chs, n_chs))
        for ix1 in range(n_chs):
            for ix2 in range(ix1 + 1, n_chs):
                conn[ix1, ix2] = pli(phase[ix1], phase[ix2])
        conn += conn.T
        conn += np.diag(np.ones(n_chs))
        conns.append(conn)
    if design == 'trial_wise':
        target_phases = np.vectorize(marker_definition.get)(events[:, 2])
    else:
        target_phases = [_get_main_target_phase(marker_definition, events)]
    df_result = pd.DataFrame({'participant': [participant] * len(conns),
                              'session': [session] * len(conns),
                              'design': [design] * len(conns),
                              'target_phase': target_phases,
                              'measure': [measure] * len(conns),
                              'value': conns})
    return df_result