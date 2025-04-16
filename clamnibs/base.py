import mne
from mne.io.brainvision.brainvision import RawBrainVision
from os.path import dirname, basename, exists, join
import numpy as np
from scipy.io import loadmat
from mne import Epochs
from mne.viz import plot_topomap
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import seaborn as sns

def _is_valid_target_phase(x):
    if isinstance(x, float):
        return (-np.pi <= x <= np.pi) or (0 <= x <= 2 * np.pi)
    elif isinstance(x, str):
        return x in ('ns', 'ol')
    return False

class RawCLAM(RawBrainVision):

    """Initialize a RawCLAM object.

    Parameters:
    -----------
    path : str
        The path to the BrainVision header file (.vhdr).
    l_freq_target : float
        Lower edge of the target frequency range.
    h_freq_target : float
        Higher edge of the target frequency range.
    tmin : float or None, optional
        For 'trial_wise' designs, this is the start time of the trial relative to the target phase marker.
        Required paramter when trigger markers are provided (e.g. marker_definition != {}).
    tmax : float or None, optional
        For 'trial_wise' designs, this is the end time of the trial relative to the target phase marker.
        Required paramter when trigger markers are provided (e.g. marker_definition != {}).
    n_chs : int, optional
        Number of EEG channels in the data (including bads).
    design : str, optional
        The experimental design type.
        'trial_wise' means that multiple phase lags were tested in the session.
        'session_wise' means that a single phase lag was tested in the session (e.g. patient treatment).
    misc_channels : list, optional
        List of miscellaneous channel names (not EEG or ECG).
    marker_definition : dict, optional
        Dictionary containing marker definitions.
        Mapping from target phase markers (e.g. 1 - 6) to target phases [-pi, pi].
        Target phases can also take the values 'ns' (no stimulation) or 'ol' (open-loop stimulation).
    sfreq : float or None, optional
        New sampling frequency, or None if the data should not be resampled.
    ignore_calibration_files: bool, optional
        If True, the user will be prompted to select bad channels and target spatial component regardless
        of the presence of calibration (.mat) files in the data folder
    default_bads : list of str, optional
        The user may specify the list of channels that are marked bad by default in viz.set_bads

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
            l_freq_target = None,
            h_freq_target = None,
            tmin = None,
            tmax = None,
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
            sfreq=None,
            ignore_calibration_files = False,
            default_bads=['Fp1', 'Fpz', 'Fp2', 'F9', 'FT9', 'TP9', 'F10', 'FT10', 'TP10']):
        super().__init__(path, preload=True)
        folder_path = dirname(path)
        self.n_chs = n_chs
        if sfreq is not None:
            self.resample(sfreq)
        freq_lims_file_path = join(folder_path, 'freq_lims.mat')
        if exists(freq_lims_file_path):
            self.l_freq_target, self.h_freq_target = loadmat(freq_lims_file_path)['freq_lims']
            print('Using target frequency range from file ({:.1f} - {:.1f} Hz)'.format(self.l_freq_target, self.h_freq_target))
        else:
            self.l_freq_target, self.h_freq_target = l_freq_target, h_freq_target
            print('Using target frequency range from parameters ({:.1f} - {:.1f} Hz)'.format(self.l_freq_target, self.h_freq_target))
        self.filter(l_freq_target, h_freq_target, picks=['envelope'])
        misc_channels = [ch for ch in misc_channels if ch in self.ch_names]
        if 'no_stim' in path.lower():
            self.is_stim = False
        else:
            self.is_stim = True
        self.design = design
        self.set_channel_types({
            **{ch: 'ecg' for ch in ecg_channels},
            **{ch: 'misc' for ch in misc_channels}})
        
        if marker_definition:
            if tmin is None or tmax is None:
                raise Exception(
                    'tmin and tmax are required parameters (cannot be None) for epoching data when a marker definition is provided')
                
        for key, value in marker_definition.items():
            if not _is_valid_target_phase(value):
                raise Exception(
                    f"""{key}:{value} is not a valid marker definition. Allowed values are either phases in the range of 
                    -π to π or 0 to 2π, or a string ('ns' for no stimulation or 'ol' for open-loop stimulation)."""
                )
        
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
        
        exclude_idx_file_path = join(folder_path, 'exclude_idx.mat')
        if exists(exclude_idx_file_path) and not ignore_calibration_files:
            exclude_idx_mat = loadmat(exclude_idx_file_path)['exclude_idx']
            if len(exclude_idx_mat) == 0:
                bads = np.array([])
            else:
                bads = np.array(self.ch_names)[exclude_idx_mat[0] - 1]
            self.info['bads'] = list(bads)
        else:
            from . import viz
            viz.set_bads(self, default_bads)
            
        p_target_file_path = join(folder_path, 'P_TARGET_{:d}.mat'.format(int(n_chs)))
        if exists(p_target_file_path) and not ignore_calibration_files:
            self.forward_full = loadmat(p_target_file_path)['P_TARGET_{:d}'.format(int(n_chs))][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no forward model was present in the data folder')
            
            from . import beamformer
            beamformer.set_forward(self, 1, 30)
            
        flip_file_path = join(folder_path, 'flip.mat')
        
        if exists(flip_file_path) and not ignore_calibration_files:
            self.flip = loadmat(flip_file_path)['flip'][0]
        else:
            if self.is_stim:
                raise Exception(
                    'Data with CLAM-tACS was loaded, but no dipole sign flip was present in the data folder')
            
            from . import beamformer
            beamformer.set_flip(self)
        events = mne.events_from_annotations(self)[0]
        if not np.all(np.isin(list(self.marker_definition.keys()), events[:, 2])):
            raise Exception('Some markers in the marker definition do not exist in the data')
            
    def plot_forward(self, sensors=False):
        from .misc import _get_ixs_goods
        from .beamformer import _get_lcmv_weights
        l_freq = self.info['highpass']
        h_freq = self.info['lowpass']
        if l_freq > 1 or h_freq < 40:
            raise Exception(
                'Forward model can only be plotted on data containing at least 1 - 40 Hz')
        ixs_good = _get_ixs_goods(self)
        info_plot = mne.pick_info(self.info, ixs_good)
        names = info_plot.ch_names if sensors else None
        forward = self.forward_full[ixs_good]
        data_broad = self.copy().filter(1, 40).get_data(ixs_good)
        COV = np.cov(data_broad)
        w = _get_lcmv_weights(COV, forward)
        psd, freqs = psd_array_welch(
            w @ data_broad, self.info['sfreq'], fmin=1, fmax=40, n_fft=int(3 * self.info['sfreq']))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1]})
        axes[0].semilogy(freqs, psd.flatten(), c='k')
        axes[0].tick_params(axis='x', labelsize=8)
        axes[0].tick_params(axis='y', labelsize=5)
        axes[0].axvline(
            self.l_freq_target, color='grey', linestyle='--', linewidth=0.5)
        axes[0].axvline(
            self.h_freq_target, color='grey', linestyle='--', linewidth=0.5)
        axes[0].set_title('Power Spectrum')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power (a.u.)')
        plot_topomap(forward, mne.pick_info(self.info, ixs_good), axes=axes[1], 
                sensors=False, contours=0, show=False)
        axes[1].set_title('Forward Model')
        sns.despine()

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
        self.flip = raw.flip
        self.n_chs = raw.n_chs