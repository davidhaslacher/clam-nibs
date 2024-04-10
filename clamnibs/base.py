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
    design : str
        The experimental design type.
        'trial_wise' means that multiple phase lags were tested in the session.
        'session_wise' means that a single phase lag was tested in the session (e.g. patient treatment).
    misc_channels : list
        List of miscellaneous channel names (not EEG or ECG).
    l_freq_target : float
        Lower edge of the target frequency range.
    h_freq_target : float
        Higher edge of the target frequency range.
    marker_definition : dict
        Dictionary containing marker definitions.
        Mapping from target phase markers (e.g. 1 - 6) to target phases [-pi, pi].
    tmin : float
        For 'trial_wise' designs, this is the start time of the trial relative to the target phase marker.
    tmax : float
        For 'trial_wise' designs, this is the end time of the trial relative to the target phase marker.

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
            design,
            misc_channels,
            l_freq_target,
            h_freq_target,
            marker_definition,
            tmin,
            tmax):
        super().__init__(path, preload=True)
        folder_path = dirname(path)
        misc_channels = [ch for ch in misc_channels if ch in self.ch_names]
        if 'stim' in path.lower():
            self.is_stim = True
        else:
            self.is_stim = False
        self.design = design
        self.set_channel_types({
            **{'ecg': 'ecg'},
            **{ch: 'misc' for ch in misc_channels}})
        self.l_freq_target = l_freq_target
        self.h_freq_target = h_freq_target
        self.marker_definition = marker_definition
        self.tmin = tmin
        self.tmax = tmax
        self.set_montage('easycap-M1', match_case=False)
        if design == 'trial_wise':
            self.participant = basename(dirname(path))
            self.session = 'T01'
        else:
            self.participant = basename(dirname(dirname(path)))
            self.session = basename(dirname(path))
        if exists('{}\\exclude_idx.mat'.format(folder_path)):
            bads = np.array(self.ch_names)[loadmat('{}\\exclude_idx.mat'.format(folder_path))['exclude_idx'][0] - 1]
            self.info['bads'] = list(bads)
        else:
            from viz import set_bads
            set_bads(self)
        if exists('{}\\P_TARGET_64.mat'.format(folder_path)):
            self.forward_64 = loadmat('{}\\P_TARGET_64.mat'.format(folder_path))['P_TARGET_64'][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no forward model was present in the data folder')
            from beamformer import set_forward
            set_forward(self, 1, 30)
        if exists('{}\\flip.mat'.format(folder_path)):
            self.flip = loadmat('{}\\flip.mat'.format(folder_path))['flip'][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no dipole sign flip was present in the data folder')
            from beamformer import set_flip
            set_flip(self)


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
    forward_64 : ndarray or None
        Forward model data for 64 channels.
    flip : integer
        Sign flip for dipole (-1 or 1).

    Raises:
    -------
    None
    """
    
    def __init__(self, raw, apply_hil):
        target_codes = list(raw.marker_definition.keys())
        events = mne.events_from_annotations(raw)[0]
        tmin = raw.tmin
        tmax = raw.tmax
        if apply_hil:
            raw_out = raw.copy().apply_hilbert(envelope=False)
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
        self.forward_64 = raw.forward_64
        self.flip = raw.flip


def get_raw(path,
            l_freq_target,
            h_freq_target,
            tmin,
            tmax,
            design='trial_wise',
            misc_channels=['envelope',
                           'envelope_am',
                           'force'],
            marker_definition={1: (0 / 6) * 2 * np.pi,
                               2: (1 / 6) * 2 * np.pi,
                               3: (2 / 6) * 2 * np.pi,
                               4: (3 / 6) * 2 * np.pi,
                               5: (4 / 6) * 2 * np.pi,
                               6: (5 / 6) * 2 * np.pi}):
    
    """Get a RawCLAM object.

    This function initializes and returns a RawCLAM object for CLAM-NIBS experiments.

    Parameters:
    -----------
    path : str
        The path to the BrainVision data file.
    design : str, optional
        The experimental design type ('trial_wise' or 'session_wise'). Default is 'trial_wise'.
    misc_channels : list, optional
        List of miscellaneous channel names. Default is ['envelope', 'envelope_am', 'force'].
    l_freq_target : float, optional
        Lower edge of the target frequency range (in Hz). Default is 4.
    h_freq_target : float, optional
        Higher edge of the target frequency range (in Hz). Default is 8.
    marker_definition : dict, optional
        Dictionary containing marker definitions.
        Mapping from target phase markers (e.g. 1 - 6) to target phases [-pi, pi].
        Default represents six equidistant target phases.

    Returns:
    --------
    raw : RawCLAM object
        A RawCLAM object initialized with the provided parameters.
    """
    raw = RawCLAM(
        path,
        design,
        misc_channels,
        l_freq_target,
        h_freq_target,
        marker_definition,
        tmin,
        tmax)
    raw.filter(l_freq_target, h_freq_target, picks=['envelope'])
    return raw

def get_epochs(raw, apply_hil=True):
    
    """Get an EpochsCLAM object from a RawCLAM object.

    This function creates an EpochsCLAM object from a RawCLAM object,
    applying the Hilbert transform by default.

    Parameters:
    -----------
    raw : RawCLAM object
        The RawCLAM object containing EEG data.
    apply_hil : bool, optional
        Whether to apply Hilbert transform to obtain analytic signal. Default is True.

    Returns:
    --------
    epochs : EpochsCLAM object
        An EpochsCLAM object initialized with the provided RawCLAM object and optional parameters.

    Raises:
    -------
    Exception:
        If the input `raw` object is not an instance of RawCLAM.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception(
            'Please use get_raw to create a RawCLAM object before using get_epochs to create an EpochsCLAM object')
    return EpochsCLAM(raw, apply_hil)