import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap
import mne
import seaborn as sns
from .misc import _get_ixs_goods
from mne.filter import next_fast_len
from scipy.signal import hilbert
from scipy.stats import binomtest
from pycircstat.descriptive import mean as circmean
from .base import RawCLAM, EpochsCLAM


def _get_lcmv_weights(COV, forward):
    COVinv = linalg.pinv(COV)
    return ((COVinv @ forward[:, None])).squeeze() / \
        (forward[None, :] @ COVinv @ forward[:, None])


def get_target(obj):
    
    """Compute the target signal from EEG data using LCMV beamforming.

    This function computes the target signal from EEG data using Linearly Constrained Minimum Variance (LCMV) beamforming [1].
    It extracts the target signal by spatially filtering the EEG data based on the forward model.

    Parameters:
    -----------
    obj : RawCLAM or EpochsCLAM object
        The RawCLAM or EpochsCLAM object containing EEG data.

    Returns:
    --------
    target : ndarray
        The target signal extracted from the EEG data using LCMV beamforming.

    Raises:
    -------
    Exception:
        If the input object is not an instance of RawCLAM or EpochsCLAM.
        
    References:
    -----------
    [1] Van Veen, Barry D., and Kevin M. Buckley. "Beamforming: A versatile approach to spatial filtering." 
        IEEE assp magazine 5.2 (1988): 4-24.
    """
    
    if not (isinstance(obj, RawCLAM) or isinstance(obj, EpochsCLAM)):
        raise Exception('get_target can only be applied to RawCLAM or EpochsCLAM objects')
    ixs_goods = _get_ixs_goods(obj)
    target_codes = obj.marker_definition.keys()
    if isinstance(obj, mne.Epochs):
        epochs_events = obj.events
        epochs_data = obj.get_data(ixs_goods)
        forward_goods = obj.forward_full[ixs_goods]
        COV = np.mean([np.cov(np.real(np.concatenate(epochs_data[epochs_events[:, 2]
                      == target_code], axis=-1))) for target_code in target_codes], axis=0)
        w = _get_lcmv_weights(COV, forward_goods)
        target = np.array([w @ ep for ep in epochs_data]).squeeze()
    else:
        raw_events = mne.events_from_annotations(obj)[0]
        raw_data = obj.get_data(ixs_goods)
        epochs = EpochsCLAM(obj)
        epochs_events = epochs.events
        epochs_data = epochs.get_data(ixs_goods)
        forward_goods = epochs.forward_full[ixs_goods]
        COV = np.mean([np.cov(np.real(np.concatenate(epochs_data[epochs_events[:, 2]
                      == target_code], axis=-1))) for target_code in target_codes], axis=0)
        w = _get_lcmv_weights(COV, forward_goods)
        target = (w @ raw_data).squeeze()
    target *= obj.flip

    return target


def set_flip(obj, plot=False):
    
    """Determine and set the dipole sign flip for the target signal.

    This method determines the dipole sign flip for the target signal based on its waveform asymmetry.
    It analyzes the phases of the target signal to determine whether the rising or falling phase is dominant (longer),
    and sets the flip factor accordingly.

    Parameters:
    -----------
    obj : RawCLAM or EpochsCLAM object
        The RawCLAM or EpochsCLAM object containing EEG data.
    plot : bool, optional
        Whether to plot the distribution of target signal phases and the result of flip determination. Default is False.

    Returns:
    --------
    None

    Raises:
    -------
    Exception:
        If the input object is not an instance of RawCLAM or EpochsCLAM.
    """
    
    if not (isinstance(obj, RawCLAM) or isinstance(obj, EpochsCLAM)):
        raise Exception('set_flip can only be applied to RawCLAM or EpochsCLAM objects')
    obj.flip = 1 # required for get_target() to run
    target = get_target(obj)
    is_complex = np.iscomplexobj(target)
    if not is_complex:
        n_times = target.shape[-1]
        n_fft = next_fast_len(n_times)
        target = hilbert(target, N=n_fft, axis=-1)[..., :n_times]
    target_phases = np.angle(target)
    n_rising = (target_phases < 0).sum()
    n_falling = (target_phases > 0).sum()
    if n_rising > n_falling:
        flip = -1
    else:
        flip = 1
    if plot:
        p = binomtest(n_rising, n_rising + n_falling).pvalue
        _, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
        ax.hist(target_phases.flatten(), color='k', alpha=0.3)
        ax.set_title('flip = {:d}, p = {:.4f}'.format(flip, p))
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.get_yaxis().set_visible(False)
        ax.axvline(circmean(target_phases.flatten()), c='r')
    obj.flip = flip


def set_forward(raw, l_freq_noise, h_freq_noise, n_comp=4):
    
    """Compute and set the forward model for target source reconstruction.

    This method computes and sets the forward model necessary for LCMV beamforming.
    A data-driven approach called Spatio-Spectral Decomposition (SSD) [1] is used to 
    find components in the signal that maximize power in the target frequency range
    while minimizing power in the noise frequency range.
    
    A component must be selected by the user by clicking on the respective power
    spectrum before closing the plot.

    Parameters:
    -----------
    raw : RawCLAM object
        The RawCLAM object containing EEG data.
    l_freq_noise : float
        Lower edge of the noise frequency range.
    h_freq_noise : float
        Higher edge of the noise frequency range.
    n_comp : int, optional
        Number of SSD components to plot. Default is 4.

    Returns:
    --------
    None

    Raises:
    -------
    Exception:
        If the input object is not an instance of RawCLAM.
        If the data does not contain frequencies of at least 1 - 40 Hz.
        If no target for stimulation (forward model) is selected by the user.
        
    References:
    -----------
    [1] Nikulin, Vadim V., Guido Nolte, and Gabriel Curio. "A novel method for reliable and fast extraction of neuronal 
        EEG/MEG oscillations on the basis of spatio-spectral decomposition." NeuroImage 55.4 (2011): 1528-1535.
    """
    
    if not isinstance(raw, RawCLAM):
        raise Exception('set_forward can only be applied to RawCLAM objects')
    sfreq = raw.info['sfreq']
    l_freq = raw.info['highpass']
    h_freq = raw.info['lowpass']
    l_freq_target = raw.l_freq_target
    h_freq_target = raw.h_freq_target
    marker_definition = raw.marker_definition
    participant = raw.participant
    ixs_goods = _get_ixs_goods(raw)
    if not (l_freq <= 1 and 40 <= h_freq):
        raise Exception(
            'Data should contain frequencies of at least 1 - 40 Hz')
    data_broad = raw.copy().filter(1, 40).get_data(ixs_goods)
    data_signal = raw.copy().filter(
        l_freq_target, h_freq_target).get_data(ixs_goods)
    data_noise = raw.copy().filter(
        l_freq_noise, h_freq_noise).get_data(ixs_goods)
    COV_B = np.cov(data_broad)
    COV_S = np.cov(data_signal)
    COV_N = np.cov(data_noise)
    evals, evecs = linalg.eig(COV_S, COV_N)
    ix = np.argsort(evals)[::-1]
    D = evecs[:, ix].T
    M = linalg.pinv(D)
    fig, axes = plt.subplots(
        n_comp, 2, figsize=(
            7, 7), gridspec_kw={
            'width_ratios': [
                1, 1]})
    for ix_comp in range(n_comp):
        w = _get_lcmv_weights(COV_B, M[:, ix_comp])
        psd, freqs = psd_array_welch(
            w @ data_broad, raw.info['sfreq'], fmin=1, fmax=40, n_fft=int(3 * raw.info['sfreq']))
        axes[ix_comp, 0].semilogy(freqs, psd.flatten(), c='k')
        axes[ix_comp, 0].tick_params(axis='x', labelsize=8)
        axes[ix_comp, 0].tick_params(axis='y', labelsize=5)
        axes[ix_comp, 0].axvline(
            l_freq_target, color='grey', linestyle='--', linewidth=0.5)
        axes[ix_comp, 0].axvline(
            h_freq_target, color='grey', linestyle='--', linewidth=0.5)
        
        plot_topomap(M[:, ix_comp], mne.pick_info(raw.info, ixs_goods), axes=axes[ix_comp,
                     1], sensors=False, contours=0, show=False)
    plt.suptitle(
        'Please select a target for stimulation by clicking anywhere inside the left plot')
    plt.figtext(0.25, 0, 'Frequency (Hz)')
    plt.figtext(0, 0.5, 'Power (a.u.)', rotation='vertical')
    plt.tight_layout()
    default_linewidth = axes[0, 0].spines['bottom'].get_linewidth()
    ix_comp = None

    def _onclick_ax(event, axes=axes[:, 0], fig=fig):
        nonlocal ix_comp
        ax_pressed = None
        ix_ax_pressed = None
        for ix, ax in enumerate(axes):
            if ax.contains(event)[0]:
                ax_pressed = ax
                ix_ax_pressed = ix
            else:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('k')
                ax.spines['left'].set_color('k')
                ax.spines['bottom'].set_linewidth(default_linewidth)
                ax.spines['left'].set_linewidth(default_linewidth)
        if ax_pressed is not None:
            ix_comp = ix_ax_pressed
            ax_pressed.spines['top'].set_visible(True)
            ax_pressed.spines['right'].set_visible(True)
            ax_pressed.spines['top'].set_color('g')
            ax_pressed.spines['right'].set_color('g')
            ax_pressed.spines['bottom'].set_color('g')
            ax_pressed.spines['left'].set_color('g')
            ax_pressed.spines['top'].set_linewidth(2 * default_linewidth)
            ax_pressed.spines['right'].set_linewidth(2 * default_linewidth)
            ax_pressed.spines['bottom'].set_linewidth(2 * default_linewidth)
            ax_pressed.spines['left'].set_linewidth(2 * default_linewidth)
            fig.canvas.draw()
    fig.canvas.mpl_connect("button_press_event", _onclick_ax)
    sns.despine()
    plt.show()
    if ix_comp is None:
        raise Exception(
            'No target for stimulation (forward model) was selected')
    forward_full = np.zeros(raw.n_chs)
    forward_full[ixs_goods] = M[:, int(ix_comp)]
    raw.forward_full = forward_full

# from scipy.io import loadmat
# raw = mne.io.read_raw_brainvision('C:\\Users\\hasla\Desktop\\rising_falling_cwm_tims\\data\\P1_DH\\calibration_no_stim.vhdr',preload=True)
# raw = raw.pick_channels(raw.ch_names[:64])
# raw.set_montage('easycap-M1',match_case=False)
# forward_model = loadmat('C:\\Users\\hasla\Desktop\\rising_falling_cwm_tims\\data\\P1_DH\\P_TARGET_64.mat')['P_TARGET_64'].squeeze()
# flip = loadmat('C:\\Users\\hasla\Desktop\\rising_falling_cwm_tims\\data\\P1_DH\\flip.mat')['flip'].squeeze()
# sfreq = raw.info['sfreq']
# mask_bad = forward_model==0
# bads = np.array(raw.ch_names)[:64][mask_bad]
# raw.drop_channels(bads)
# get_forward(raw,8,14,1,40)