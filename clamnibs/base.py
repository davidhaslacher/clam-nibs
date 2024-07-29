import mne
from mne.io.brainvision.brainvision import RawBrainVision
from os.path import dirname, basename, exists
import numpy as np
from scipy.io import loadmat
from mne import Epochs

class RawCLAM(RawBrainVision):

    """Initialize a RawCLAM object.

    Parameters:
    -----------
    path : str
        The path to the BrainVision data file.
    l_freq_target : float
        Lower edge of the target frequency range.
    h_freq_target : float
        Higher edge of the target frequency range.
    tmin : float
        For 'trial_wise' designs, this is the start time of the trial relative to the target phase marker.
    tmax : float
        For 'trial_wise' designs, this is the end time of the trial relative to the target phase marker.
    n_chs : int
        Number of EEG channels in the data (including bads).
    design : str
        The experimental design type.
        'trial_wise' means that multiple phase lags were tested in the session.
        'session_wise' means that a single phase lag was tested in the session (e.g. patient treatment).
    misc_channels : list
        List of miscellaneous channel names (not EEG or ECG).
    marker_definition : dict
        Dictionary containing marker definitions.
        Mapping from target phase markers (e.g. 1 - 6) to target phases [-pi, pi].
    sfreq : float or None
        New sampling frequency, or None if the data should not be resampled.

    Notes:
    ------
    This class extends RawBrainVision and provides additional functionality specific to CLAM-NIBS experiments.

    Raises:
    -------
    Exception:
        - If data with CLAM-NIBS is loaded but no forward model is present in the data folder.
        - If data with CLAM-NIBS is loaded but no dipole sign flip is present in the data folder.
    """

    def __init__(
            self,
            path,
            l_freq_target,
            h_freq_target,
            tmin,
            tmax,
            n_chs=64,
            design='trial_wise',
            ecg_channels=[],
            misc_channels=['envelope',
                           'envelope_am'],
            marker_definition={1: (0 / 6) * 2 * np.pi,
                               2: (1 / 6) * 2 * np.pi,
                               3: (2 / 6) * 2 * np.pi,
                               4: (3 / 6) * 2 * np.pi,
                               5: (4 / 6) * 2 * np.pi,
                               6: (5 / 6) * 2 * np.pi},
            sfreq=None):
        super().__init__(path, preload=True)
        self.n_chs = n_chs
        if sfreq is not None:
            self.resample(sfreq)
        self.filter(l_freq_target, h_freq_target, picks=['envelope'])
        folder_path = dirname(path)
        misc_channels = [ch for ch in misc_channels if ch in self.ch_names]
        if 'no_stim' in path.lower():
            self.is_stim = False
        else:
            self.is_stim = True
        self.design = design
        self.set_channel_types({
            **{ch: 'ecg' for ch in ecg_channels},
            **{ch: 'misc' for ch in misc_channels}})
        self.l_freq_target = l_freq_target
        self.h_freq_target = h_freq_target
        self.marker_definition = marker_definition
        self.tmin = tmin
        self.tmax = tmax
        self.set_montage('easycap-M1', match_case=False, on_missing='warn')
        if design == 'trial_wise':
            self.participant = basename(dirname(path))
            self.session = 'T01'
        else:
            self.participant = basename(dirname(dirname(path)))
            self.session = basename(dirname(path))
        
        exclude_idx_file_path = '{}\\exclude_idx.mat'.format(folder_path)
        if exists(exclude_idx_file_path):
            bads = np.array(self.ch_names)[loadmat(exclude_idx_file_path)['exclude_idx'][0] - 1]
            self.info['bads'] = list(bads)
        else:
            from . import viz
            viz.set_bads(self)
            
        p_target_file_path = '{}\\P_TARGET_{:d}.mat'.format(folder_path, int(n_chs))
        if exists(p_target_file_path):
            self.forward_full = loadmat(p_target_file_path)['P_TARGET_{:d}'.format(int(n_chs))][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no forward model was present in the data folder')
            
            from . import beamformer
            beamformer.set_forward(self, 1, 30)
            
        flip_file_path = '{}\\flip.mat'.format(folder_path)
        if exists(flip_file_path):
            self.flip = loadmat(flip_file_path)['flip'][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no dipole sign flip was present in the data folder')
            
            from . import beamformer
            beamformer.set_flip(self)
        


class EpochsCLAM(Epochs):
    
    """Initialize an EpochsCLAM object.

    Parameters:
    -----------
    raw : RawCLAM object
        The EEG data.
    apply_hil : bool
        Whether to apply Hilbert transformation to the EEG data to obtain the analytic signal.

    Notes:
    ------
    This class extends Epochs and provides additional functionality specific to CLAM-NIBS experiments.

    Attributes:
    -----------
    design : str
        The experimental design type ('trial_wise' or 'session_wise').
    l_freq_target : float
        Lower edge of the target frequency range (in Hz).
    h_freq_target : float
        Higher edge of the target frequency range (in Hz).
    marker_definition : dict
        Dictionary containing marker definitions.
        Mapping from target phase markers (e.g. 1 - 6) to target phases [-pi, pi].
    is_stim : bool
        Indicates whether the data was recorded in the presence of CLAM-NIBS or not.
    participant : str
        Participant identifier.
    session : str
        Session identifier.
    forward_full : ndarray or None
        Forward model for all EEG channels (with zero for bads).
    flip : integer
        Sign flip for dipole (-1 or 1).

    Raises:
    -------
    None
    """
    
    def __init__(self, raw, apply_hil=True):
        if not isinstance(raw, RawCLAM):
            raise Exception(
                'Please use get_raw to create a RawCLAM object before using get_epochs to create an EpochsCLAM object')
        target_codes = list(raw.marker_definition.keys())
        events = mne.events_from_annotations(raw)[0]
        tmin = raw.tmin
        tmax = raw.tmax
        if apply_hil:
            picks = [ch_name for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type=='eeg']+['envelope']
            raw_out = raw.copy().apply_hilbert(picks=picks, envelope=False)
        else:
            raw_out = raw
        super().__init__(
            raw_out,
            events,
            event_id=target_codes,
            on_missing='ignore',
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            proj=False,
            preload=True)
        self.design = raw.design
        self.l_freq_target = raw.l_freq_target
        self.h_freq_target = raw.h_freq_target
        self.marker_definition = raw.marker_definition
        self.is_stim = raw.is_stim
        self.participant = raw.participant
        self.session = raw.session
        self.forward_full = raw.forward_full
        self.n_chs = raw.n_chs