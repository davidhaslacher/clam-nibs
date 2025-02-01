import numpy as np
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from scipy.sparse import coo_matrix
import scipy
from mne.stats import ttest_ind_no_p
from mne.viz.topomap import _get_pos_outlines
from mne.utils.check import _check_sphere
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.viz import plot_topomap
from functools import partial
import pandas as pd
from scipy.stats import ttest_ind
from numpy.random import permutation
from scipy.stats import permutation_test
import os
import matplotlib
from fooof import FOOOF
from fooof.analysis import get_band_peak_fm
from .misc import _fmt
from .misc import df_to_array

# TODO:
# Statistics over sessions in session_wise studies

def _dft(x, plot_sine=False):
    n_bins = len(x)
    phases = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    c = (x*np.exp(-1j*phases)).sum()*2/n_bins
    amp, phase = np.abs(c), np.angle(c)
    if plot_sine:
        n_bins = len(x)
        xs = np.linspace(-0.5, n_bins-0.5, 50)
        phases_xs = xs*2*np.pi/n_bins
        dy = np.nanmean(x)
        phases = np.linspace(phases_xs[0], phases_xs[-1],50)
        ys = amp*np.cos(phases + phase)+dy
        plt.plot(xs, ys, c='k', linewidth=3, zorder=2)
    return amp, _wrap(-phase)

# Each arg in args should be of shape (n_samples, n_features)
# This function averages over n_samples and computes DFT amplitude for each feature 
from joblib import Parallel, delayed
def _vectorized_dft_amp(*args):
    if args[0].ndim == 1:
        args = [arg[:, None] for arg in args]
    # Compute mean across samples for each feature
    means = np.array([np.nanmean(arg, axis=0) for arg in args])
    # Use parallel processing to compute DFT amplitudes
    amps = Parallel(n_jobs=-1)(delayed(_dft)(means[:, ix_feature]) for ix_feature in range(means.shape[1]))
    # Extract amplitude from the result
    amps = np.array([amp[0] for amp in amps])
    return amps

def _dft_amp_stat(*args):
    return _vectorized_dft_amp(*args)

# This is a hacky solution to make permutation_cluster_test compatible with the SINE FIT BINNED procedure outlined in [1].
# Unfortunately, permutation_cluster_test can't handle trial-level data for multiple participants.
# Therefore, args are ignored and orig_args are used here.
# This function permutes trials across phase bins within each participant.
# Trials are averaged within each phase bin and the DFT amplitude is computed for each participant.
# The group-level test statistic is then the DFT amplitude averaged over participants.
# [1] Zoefel, Benedikt, et al. "How to test for phasic modulation of neural and behavioural responses." Neuroimage 202 (2019): 116175.
# orig_args is a list of arrays, one per participant i, each of shape (n_phases, n_epochs_i, n_conns)
def _dft_amp_stat_group(*args, orig_args=None):
    global first_pass_done
    # Shuffle if original test statistic was computed (first pass done)
    if first_pass_done:
        for ix_arg in range(len(orig_args)):
            n_phases = orig_args[ix_arg].shape[0]
            n_epochs = orig_args[ix_arg].shape[1]
            ix0, ix1 = np.unravel_index(np.random.permutation(n_phases * n_epochs),
                                                              (n_phases, n_epochs))
            orig_args[ix_arg] = orig_args[ix_arg][ix0, ix1, :]
            orig_args[ix_arg] = orig_args[ix_arg].reshape(n_phases, n_epochs, -1)
    else:
        first_pass_done = True
    all_dft_amps = []
    for orig_arg in orig_args:
        dft_amps = _vectorized_dft_amp(*orig_arg)
        all_dft_amps.append(dft_amps)
    return np.mean(all_dft_amps, axis=0)

def _wrap(phases):
    return np.angle(np.exp(1j*phases))

def test_sensor_network_modulation(
        df_data,
        info,
        test_level='participant',
        measure='phase_lag_index',
        threshold_percentile=95):
    
    """Test phase-dependent modulation of functional connectivity at the sensor level.

    This function performs a network-based permutation test to identify networks whose
    connectivity is modulated by CLAM-NIBS, and plots the results.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The DataFrame containing connectivity matrices per participant, or per epoch
        for each participant.
    
    info : mne.Info
        The Info object for topographic plotting.
        
    test_level : str, optional (default='participant')
        The level at which the test should be performed ('participant' or 'group').
        
    measure : str, optional (default='phase_lag_index')
        The connectivity measure employed.
        
    plot : str or bool, optional (default=False)
        If string, the plot(s) will be saved at that path.
        If True, a figure will be created but not shown or saved.
        If False, no figure will be created.

    Raises:
    -------
    Exception:
        If the test level is not \'participant\' or \'group\'
    """
    
    df_data = df_data[df_data['measure'] == measure]
    if test_level == 'participant':
        df_results = _test_sensor_network_modulation_participant(df_data, 
                                                                 info, 
                                                                 measure,
                                                                 threshold_percentile)
        return df_results
    elif test_level == 'group':
        df_results = _test_sensor_network_modulation_group(df_data, 
                                                           info, 
                                                           measure, 
                                                           threshold_percentile)
        return df_results
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_sensor_network_modulation_participant(df_data, info, measure, threshold_percentile):
    gb_participant = df_data.groupby('participant')
    df_results = pd.DataFrame()
    for participant, df_participant in gb_participant:
        gb_target_phases = df_participant.sort_values(
            'target_phase').groupby('target_phase')
        target_phases = []
        data = []
        for target_phase, df_target_phase in gb_target_phases:
            target_phases.append(target_phase)
            data.append(np.stack(df_target_phase['value']))
        n_obs = data[0].shape[0]
        n_chs = data[0].shape[1]
        n_conns = n_chs**2
        data = [d.reshape(-1, n_conns) for d in data]
        adjacency = np.zeros((n_conns, n_conns))
        for ix_conn_1 in range(n_conns):
            for ix_conn_2 in range(ix_conn_1 + 1, n_conns):
                ix_ch_1 = ix_conn_1 // n_chs
                ix_ch_2 = ix_conn_1 % n_chs
                ix_ch_3 = ix_conn_2 // n_chs
                ix_ch_4 = ix_conn_2 % n_chs
                if not {ix_ch_1, ix_ch_2}.isdisjoint({ix_ch_3, ix_ch_4}):
                    adjacency[ix_conn_1, ix_conn_2] = 1
        adjacency += adjacency.T
        adjacency = coo_matrix(adjacency)
        if len(data) == 2:
            # use t-statistic
            threshold = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_obs * 2 - 2)
            tvals, clusters, pvals, _ = permutation_cluster_test(data,
                                                                 threshold=threshold,
                                                                 adjacency=adjacency,
                                                                 out_type='indices',
                                                                 step_down_p=0,
                                                                 t_power=1,
                                                                 stat_fun=ttest_ind_no_p,
                                                                 tail=0,
                                                                 n_jobs=1,
                                                                 verbose=True)
            tvals_unit = 'ttest_ind'
        else:
            stat_fun = _dft_amp_stat
            threshold = np.nanpercentile(
                [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_conns)], threshold_percentile)
            tvals, clusters, pvals, _ = permutation_cluster_test(data,
                                                                 threshold=threshold,
                                                                 adjacency=adjacency,
                                                                 out_type='indices',
                                                                 step_down_p=0,
                                                                 t_power=1,
                                                                 stat_fun=stat_fun,
                                                                 tail=0,
                                                                 n_jobs=1,
                                                                 verbose=True)
            tvals_unit = 'dft_amp'
        tvals_sig = []
        pvals_sig = []
        conns_sig = []
        for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
            if pval < 0.05:
                tvals_sig.append(tvals[cluster])
                pvals_sig.append(pval)
                conns_cluster = []
                for ix_conn in cluster[0]:
                    ix_ch_1 = ix_conn // n_chs
                    ix_ch_2 = ix_conn % n_chs
                    conns_cluster.append([ix_ch_1, ix_ch_2])
                conns_sig.append(conns_cluster)
        df_append = pd.DataFrame(
            {
                'participant': [participant] *
                len(tvals_sig),
                't_unit': [tvals_unit] *
                len(tvals_sig),
                't_values': tvals_sig,
                'p_value': pvals_sig,
                'connections': conns_sig})
        df_results = pd.concat([df_results, df_append])
    return df_results


def _test_sensor_network_modulation_group(df_data, info, measure, threshold_percentile):
    all_data = []
    gb_participants = df_data.groupby('participant')
    for participant, df_participant in gb_participants:
        gb_target_phases = df_participant.sort_values(
            'target_phase').groupby('target_phase')
        target_phases = []
        data = []
        for target_phase, df_target_phase in gb_target_phases:
            target_phases.append(target_phase)
            data.append(np.stack(df_target_phase['value']))
        n_epochs = np.min([d.shape[0] for d in data])
        data = np.array([d[:n_epochs] for d in data])
        # data is now (n_phases, n_epochs, n_chs, n_chs)
        all_data.append(data)
    n_participants = len(all_data)
    n_phases = data[0].shape[0]
    n_chs = data[0].shape[-1]
    n_conns = n_chs**2
    all_data = [d.reshape(n_phases, -1, n_conns) for d in all_data]
    # each data in all_data is now (n_phases, n_epochs, n_conns)
    adjacency = np.zeros((n_conns, n_conns))
    for ix_conn_1 in range(n_conns):
        for ix_conn_2 in range(ix_conn_1 + 1, n_conns):
            ix_ch_1 = ix_conn_1 // n_chs
            ix_ch_2 = ix_conn_1 % n_chs
            ix_ch_3 = ix_conn_2 // n_chs
            ix_ch_4 = ix_conn_2 % n_chs
            if not {ix_ch_1, ix_ch_2}.isdisjoint({ix_ch_3, ix_ch_4}):
                adjacency[ix_conn_1, ix_conn_2] = 1
    adjacency += adjacency.T
    adjacency = coo_matrix(adjacency)
    # Unfortunately, permutation_cluster_test can't handle single-trial data for multiple participants.
    # Therefore, we will pass dummy data of the expected shape here and implement the permutation and 
    # test statistic computation in _dft_amp_stat_group. See _dft_amp_stat_group for more information.
    global first_pass_done
    first_pass_done = False
    stat_fun = partial(
        _dft_amp_stat_group,
        orig_args=all_data)
    dummy_data = [np.empty((n_participants, n_conns)) for ix in range(n_phases)]
    threshold = np.nanpercentile(stat_fun(dummy_data), threshold_percentile)
    # Hacky, have to set this to false again, because it is used in permutation_cluster_test
    first_pass_done = False
    tvals, clusters, pvals, _ = permutation_cluster_test(dummy_data,
                                                            threshold=threshold,
                                                            n_permutations=100,
                                                            adjacency=adjacency,
                                                            out_type='indices',
                                                            step_down_p=0,
                                                            t_power=1,
                                                            stat_fun=stat_fun,
                                                            tail=0,
                                                            n_jobs=1,
                                                            verbose=True,
                                                            buffer_size=n_conns)
    print('Found {:d} clusters'.format(len(clusters)))
    print('p-values: ', pvals)
    tvals_unit = 'dft_amp'
    tvals_sig = []
    pvals_sig = []
    conns_sig = []
    for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
        if pval < 0.05:
            tvals_sig.append(tvals[cluster])
            pvals_sig.append(pval)
            conns_cluster = []
            for ix_conn in cluster[0]:
                ix_ch_1 = ix_conn // n_chs
                ix_ch_2 = ix_conn % n_chs
                conns_cluster.append([ix_ch_1, ix_ch_2])
            conns_sig.append(conns_cluster)
    df_results = pd.DataFrame({'participant': ['group'] * len(tvals_sig),
                              't_unit': [tvals_unit] * len(tvals_sig),
                              't_values': tvals_sig,
                              'p_value': pvals_sig,
                              'connections': conns_sig})
    return df_results


def test_sensor_cluster_modulation(
        df_data,
        info,
        test_level='participant',
        measure='amplitude',
        threshold_percentile=95,
        plot=False):
    
    """Test phase-dependent modulation of arbitrary outcome measure (e.g. power) at the sensor level.

    This function performs a cluster-based permutation test to identify clusters of sensors whose
    value of the outcome measure is modulated by CLAM-NIBS, and plots the results.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The DataFrame containing outcome measure (for all sensors) per participant, or per epoch
        for each participant.
    
    info : mne.Info
        The Info object for topographic plotting.
        
    test_level : str, optional (default='participant')
        The level at which the test should be performed ('participant' or 'group').
        
    measure : str, optional (default='amplitude')
        The outcome measure employed.
        
    plot : str or bool, optional (default=False)
        If string, the plot(s) will be saved at that path.
        If True, a figure will be created but not shown or saved.
        If False, no figure will be created.

    Raises:
    -------
    Exception:
        If the test level is not \'participant\' or \'group\'
    """
    
    df_data = df_data[df_data['measure'] == measure]
    if test_level == 'participant':
        df_results = _test_sensor_cluster_modulation_participant(df_data, 
                                                                 info, 
                                                                 measure,
                                                                 threshold_percentile, 
                                                                 plot)
        return df_results
    elif test_level == 'group':
        df_results = _test_sensor_cluster_modulation_group(df_data, 
                                                           info, 
                                                           measure, 
                                                           threshold_percentile,
                                                           plot)
        return df_results
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_sensor_cluster_modulation_participant(df_data, info, measure, threshold_percentile, plot):
    gb_participant = df_data.groupby('participant')
    df_results = pd.DataFrame()
    for participant, df_participant in gb_participant:
        gb_target_phases = df_participant.sort_values(
            'target_phase').groupby('target_phase')
        target_phases = []
        data = []
        for target_phase, df_target_phase in gb_target_phases:
            target_phases.append(target_phase)
            data.append(np.stack(df_target_phase['value']))
        n_obs = data[0].shape[0]
        n_chs = data[0].shape[1]
        adjacency = mne.channels.find_ch_adjacency(info, None)[0]
        if len(data) == 2:
            # use t-statistic
            threshold = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_obs * 2 - 2)
            tvals, clusters, pvals, _ = permutation_cluster_test(data,
                                                                 threshold=threshold,
                                                                 adjacency=adjacency,
                                                                 out_type='indices',
                                                                 step_down_p=0,
                                                                 t_power=1,
                                                                 stat_fun=ttest_ind_no_p,
                                                                 tail=0,
                                                                 n_jobs=1,
                                                                 verbose=True)
            tvals_unit = 'ttest_ind'
        else:
            stat_fun = _dft_amp_stat
            threshold = np.nanpercentile(
                [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_chs)], threshold_percentile)
            tvals, clusters, pvals, _ = permutation_cluster_test(data,
                                                                 threshold=threshold,
                                                                 adjacency=adjacency,
                                                                 out_type='indices',
                                                                 step_down_p=0,
                                                                 t_power=1,
                                                                 stat_fun=stat_fun,
                                                                 tail=0,
                                                                 n_jobs=1,
                                                                 verbose=True)
            tvals_unit = 'dft_amp'
        tvals_sig = []
        pvals_sig = []
        channels_sig = []
        mask_sig = np.zeros(n_chs).astype(bool)
        for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
            if pval < 0.05:
                tvals_sig.append(tvals[cluster])
                pvals_sig.append(pval)
                channels_sig.append(info.ch_names[cluster])
                mask_sig[cluster] = True
        if plot:
            im = plot_topomap(
                tvals,
                info,
                sensors=False,
                mask=mask_sig,
                contours=0)[0]
            cbar = plt.colorbar(im)
            cbar.set_label(
                'Modulation of {} \n ({})'.format(
                    _fmt(measure), _fmt(tvals_unit)))
            plt.title(
                '{}, Modulation of {}'.format(
                    participant,
                    _fmt(measure)))
            plt.tight_layout()
        df_append = pd.DataFrame(
            {
                'participant': [participant] *
                len(tvals_sig),
                't_unit': [tvals_unit] *
                len(tvals_sig),
                't_values': tvals_sig,
                'p_value': pvals_sig,
                'channels': channels_sig})
        df_results = pd.concat([df_results, df_append])
    return df_results


def _test_sensor_cluster_modulation_group(df_data, info, measure, threshold_percentile, plot):
    all_data = []
    gb_participants = df_data.groupby('participant')
    for participant, df_participant in gb_participants:
        gb_target_phases = df_participant.sort_values(
            'target_phase').groupby('target_phase')
        target_phases = []
        data = []
        for target_phase, df_target_phase in gb_target_phases:
            target_phases.append(target_phase)
            data.append(np.stack(df_target_phase['value']))
        n_epochs = np.min([d.shape[0] for d in data])
        data = np.array([d[:n_epochs] for d in data])
        # data is now (n_phases, n_epochs, n_chs)
        all_data.append(data)
    adjacency = mne.channels.find_ch_adjacency(info, None)[0]
    n_participants = len(all_data)
    n_phases = all_data[0].shape[0]
    n_chs = all_data[0].shape[2]
    # Unfortunately, permutation_cluster_test can't handle single-trial data for multiple participants.
    # Therefore, we will pass dummy data of the expected shape here and implement the permutation and 
    # test statistic computation in _dft_amp_stat_group. See _dft_amp_stat_group for more information.
    global first_pass_done
    first_pass_done = False
    stat_fun = partial(
        _dft_amp_stat_group,
        orig_args=all_data)
    dummy_data = [np.empty((n_participants, n_chs)) for ix in range(n_phases)]
    threshold = np.nanpercentile(stat_fun(dummy_data), threshold_percentile)
    # Hacky, have to set this to false again, because it is used in permutation_cluster_test
    first_pass_done = False
    tvals, clusters, pvals, _ = permutation_cluster_test(dummy_data,
                                                            threshold=threshold,
                                                            n_permutations=100,
                                                            adjacency=adjacency,
                                                            out_type='indices',
                                                            step_down_p=0,
                                                            t_power=1,
                                                            stat_fun=stat_fun,
                                                            tail=0,
                                                            n_jobs=1,
                                                            verbose=True,
                                                            buffer_size=n_chs)
    print('Found {:d} clusters'.format(len(clusters)))
    print('p-values: ', pvals)
    tvals_unit = 'dft_amp'
    tvals_sig = []
    pvals_sig = []
    channels_sig = []
    mask_sig = np.zeros(n_chs).astype(bool)
    for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
        if pval < 0.05:
            tvals_sig.append(tvals[cluster])
            pvals_sig.append(pval)
            channels_sig.append(info.ch_names[cluster])
            mask_sig[cluster] = True
    if plot:
        im = plot_topomap(
            tvals,
            info,
            sensors=False,
            mask=mask_sig,
            contours=0)[0]
        cbar = plt.colorbar(im)
        cbar.set_label(
            'Modulation of {} \n ({})'.format(
                _fmt(measure),
                _fmt(tvals_unit)))
        plt.title('Group, Modulation of {}'.format(_fmt(measure)))
        plt.tight_layout()
    df_results = pd.DataFrame({'participant': ['group'] * len(tvals_sig),
                              't_unit': [tvals_unit] * len(tvals_sig),
                              't_values': tvals_sig,
                              'p_value': pvals_sig,
                              'channels': channels_sig})
    return df_results


def test_modulation(
        df_data,
        test_level='participant',
        measure='amplitude',
        agg_func=np.nanmean,
        plot=False,
        plot_mode='box_strip'):
    
    """Test phase-dependent modulation of arbitrary outcome measure (e.g. amplitude).

    This function performs a permutation test to identify phase-dependent modulation
    of the outcome measure by CLAM-NIBS, and plots the results.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The DataFrame containing outcome measure per participant, or per epoch
        for each participant.
        
    test_level : str, optional (default='participant')
        The level at which the test should be performed ('participant' or 'group').
        
    measure : str, optional (default='amplitude')
        The outcome measure employed.
        
    agg_func : callable, optimal (default=np.nanmean)
        The function used to aggregate across trials within each phase bin (e.g. np.nanmean or np.nanstd).
        
    plot : str or bool, optional (default=False)
        If string, the plot(s) will be saved at that path.
        If True, a figure will be created but not shown or saved.
        If False, no figure will be created.
        
    plot_mode : str, optional (default='box_strip')
        The plotting mode. If all data points should be shown, use 'box_strip' for a boxplot and stripplot.
        If data points should only be visualized as aggregated measures (e.g. accuracy across all trials or 
        variability across all ECG RR intervals), use 'bar' for a barplot.

    Raises:
    -------
    Exception:
        If the test level is not \'participant\' or \'group\'
    """
    
    if np.any(np.isin(['ol', 'ns'], df_data['target_phase'])):
        raise Exception('test_modulation test for phase-dependent modulation, it currently \
                        does not support open-loop or no stimulation conditions')
    df_data = df_data[df_data['measure'] == measure]
    if test_level == 'participant':
        df_results = _test_modulation_participant(df_data=df_data, measure=measure, agg_func=agg_func, plot=plot, plot_mode=plot_mode)
        return df_results
    elif test_level == 'group':
        df_results = _test_modulation_group(df_data=df_data, measure=measure, agg_func=np.nanmean, plot=plot, plot_mode=plot_mode)
        return df_results
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_modulation_participant(df_data, measure, agg_func, plot, plot_mode):
    df_results = pd.DataFrame()
    participants = df_data['participant'].unique()
    for participant in participants:
        df_participant = df_data[df_data['participant'] == participant]
        gb_target_phase = df_participant.groupby(by='target_phase')
        target_phases = []
        target_measures = []
        for target_phase, df_target_phase in gb_target_phase:
            target_phases.append(target_phase)
            target_measures.append(df_target_phase['value'].to_numpy())
        if len(target_phases) == 2:
            tval, pval = ttest_ind(target_measures[0], target_measures[1])
            tval_unit = 'ttest_ind'
        else:
            res = permutation_test(target_measures,
                                   lambda *x: _dft(np.array([agg_func(x_) for x_ in x]))[0],
                                   permutation_type='independent',
                                   alternative='greater',
                                   n_resamples=1000)
            tval = res.statistic
            pval = res.pvalue
            tval_unit = 'dft_amp'
        if plot:
            plt.figure()
            x = np.concatenate([[target_phases[ix]] * len(target_measures[ix])
                                for ix in range(len(target_phases))])
            x = [round(np.rad2deg(ph)) for ph in x]
            y = np.concatenate(target_measures)
            df_plot = pd.DataFrame(
                {'Target Phase (°)': x, '{}'.format(measure): y})
            df_plot_agg = df_plot.sort_values('Target Phase (°)').groupby('Target Phase (°)') \
                            .agg({'{}'.format(measure) : agg_func}).reset_index()
            if plot_mode == 'box_strip':
                sns.boxplot(
                    df_plot,
                    x='Target Phase (°)',
                    y='{}'.format(measure),
                    color='k',
                    boxprops=dict(
                        alpha=0.5),
                    showmeans=True,
                    zorder=0,
                    showfliers=False)
                sns.stripplot(
                    df_plot,
                    x='Target Phase (°)',
                    y='{}'.format(measure),
                    color='r',
                    alpha=0.8,
                    zorder=1)
            elif plot_mode == 'bar':
                sns.barplot(
                    df_plot_agg,
                    x='Target Phase (°)',
                    y='{}'.format(measure),
                    color='k',
                    alpha=0.5,
                    errorbar=None,
                    zorder=0)
            if len(target_phases) == 2:
                plt.title('t = {:.3e}, p = {:.3e}'.format(tval, pval))
            else:
                avgs = df_plot_agg['{}'.format(measure)].to_numpy()
                _dft(avgs, plot_sine=True)
                plt.title('dft_amp = {:.3e}, p = {:.3e}'.format(tval, pval))
            sns.despine()
            if isinstance(plot, str):
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42
                sns.set_context('paper')
                if not os.path.exists(plot):
                    os.makedirs(plot,exist_ok=True)
                plt.savefig('{}_modulation_{}.pdf'.format(measure, participant))
                plt.close()
        df_append = pd.DataFrame({'participant': [participant],
                                  't_unit': [tval_unit],
                                  't_value': [tval],
                                  'p_value': [pval]})
        df_results = pd.concat([df_results, df_append])
    return df_results


def _test_modulation_group(df_data, measure, agg_func, plot, plot_mode):
    gb_target_phase = df_data.groupby('target_phase')
    target_phases = []
    target_measures = []
    for target_phase, df_target_phase in gb_target_phase:
        target_phases.append(target_phase)
        target_measures.append(df_target_phase['value'].to_numpy())
    if len(target_phases) == 2:
        tval, pval = ttest_ind(target_measures[0], target_measures[1])
        tval_unit = 'ttest_ind'
    else:
        res = permutation_test(target_measures,
                                lambda *x: _dft(np.array([agg_func(x_) for x_ in x]))[0],
                                permutation_type='samples',
                                alternative='greater',
                                n_resamples=1000)
        tval = res.statistic
        pval = res.pvalue
        tval_unit = 'dft_amp'
    if plot:
        x = np.concatenate([[target_phases[ix]] * len(target_measures[ix])
                            for ix in range(len(target_phases))])
        x = [round(np.rad2deg(ph)) for ph in x]
        y = np.concatenate(target_measures)
        df_plot = pd.DataFrame(
            {'Target Phase (°)': x, '{}'.format(measure): y})
        df_plot_agg = df_plot.sort_values('Target Phase (°)').groupby('Target Phase (°)') \
                .agg({'{}'.format(measure) : agg_func}).reset_index()
        if plot_mode == 'box_strip':
            sns.boxplot(
                df_plot,
                x='Target Phase (°)',
                y='{}'.format(measure),
                color='k',
                boxprops=dict(
                    alpha=.5),
                showmeans=True,
                zorder=0,
                showfliers=False)
            sns.stripplot(
                df_plot,
                x='Target Phase (°)',
                y='{}'.format(measure),
                color='r',
                alpha=0.8,
                zorder=1)
        elif plot_mode == 'bar':
            sns.barplot(
                df_plot_agg,
                x='Target Phase (°)',
                y='{}'.format(measure),
                color='k',
                alpha=0.5,
                errorbar=None,
                zorder=0)
        if len(target_phases) == 2:
            plt.title('t = {:.3e}, p = {:.3e}'.format(tval, pval))
        else:
            avgs = df_plot_agg['{}'.format(measure)].to_numpy()
            _dft(avgs, plot_sine=True)
            plt.title('dft_amp = {:.3e}, p = {:.3e}'.format(tval, pval))
        sns.despine()
        if isinstance(plot, str):
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            sns.set_context('paper')
            if not os.path.exists(plot):
                os.makedirs(plot,exist_ok=True)
            plt.savefig('{}_modulation_group.pdf'.format(measure))
            plt.close()
    df_results = pd.DataFrame({'participant': ['group'],
                              't_unit': [tval_unit],
                              't_value': [tval],
                              'p_value': [pval]})
    return df_results

def _group_mean_psds(group):
    stacked = np.vstack(group['value'])
    mean = np.mean(stacked, axis=0)
    return pd.Series([mean])

def test_modulation_psd(
        df_data,
        test_level='participant',
        measure='power',
        plot=False,
        plot_mode='box_strip'):
    
    """Test phase-dependent modulation of amplitude, frequency, or aperiodic exponent from power spectral density.

    This function performs a permutation test to identify phase-dependent modulation
    of the outcome measure by CLAM-NIBS, and plots the results.

    Parameters:
    -----------
    df_data : pandas.DataFrame
        The DataFrame containing outcome measure per participant, or per epoch
        for each participant.
        
    test_level : str, optional (default='participant')
        The level at which the test should be performed ('participant' or 'group').
        
    measure : str, optional (default='power')
        The outcome measure employed ('power', 'frequency', or 'aperiodic').
        
    plot : str or bool, optional (default=False)
        If string, the plot(s) will be saved at that path.
        If True, a figure will be created but not shown or saved.
        If False, no figure will be created.
        
    plot_mode : str, optional (default='box_strip')
        The plotting mode. If all data points should be shown, use 'box_strip' for a boxplot and stripplot.
        If data points should only be visualized as aggregated measures (e.g. accuracy across all trials or 
        variability across all ECG RR intervals), use 'bar' for a barplot.

    Raises:
    -------
    Exception:
        If the test level is not \'participant\' or \'group\'
    """
    
    if np.any(np.isin(['ol', 'ns'], df_data['target_phase'])):
        raise Exception('test_modulation test for phase-dependent modulation, it currently \
                        does not support open-loop or no stimulation conditions')
    if measure not in ['power', 'frequency', 'aperiodic']:
        raise Exception('Only power, frequency, and aperiodic exponent are supported')
    df_data = df_data[df_data['measure'] == 'psd']
    if test_level == 'participant':
        df_results = _test_modulation_participant_psd(df_data=df_data, measure=measure, plot=plot)
        return df_results
    elif test_level == 'group':
        df_results = _test_modulation_psd_group(df_data=df_data, measure=measure, plot=plot)
        return df_results
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')
        
def _fooof_agg(x, measure, freqs, l_freq_target, h_freq_target):
    x = [x_.mean(axis=0) for x_ in x]
    agg_measures = []
    for psd in x:
        fm = FOOOF(verbose=False)
        fm.fit(freqs, psd)
        freq, pow = get_band_peak_fm(fm, [l_freq_target, h_freq_target], select_highest=True)[:2]
        aper = fm.get_results().aperiodic_params[-1]
        if measure == 'power':
            agg_measures.append(pow)
        elif measure == 'frequency':
            agg_measures.append(freq)
        elif measure == 'aperiodic':
            agg_measures.append(aper)
            
        # fm.plot()
        # plt.axvline(freq, color='r', linestyle='--')
        # plt.title('{:.2f} Hz, {:.2f} a.u.'.format(freq, pow))
        # plt.show()
            
    return agg_measures

def _fooof_stat(x, measure, freqs, l_freq_target, h_freq_target, stat):
    agg_measures = _fooof_agg(x, measure, freqs, l_freq_target, h_freq_target)
    if stat == 'ttest':
        return agg_measures[0] - agg_measures[1]
    elif stat == 'dft':
        return _dft(agg_measures)[0]

def _test_modulation_participant_psd(df_data, measure, plot):
    df_results = pd.DataFrame()
    participants = df_data['participant'].unique()
    for participant in participants:
        df_participant = df_data[df_data['participant'] == participant]
        gb_target_phase = df_participant.groupby('target_phase')
        target_phases = []
        target_psds = []
        for target_phase, df_target_phase in gb_target_phase:
            target_phases.append(target_phase)
            target_psds.append(df_target_phase['value'].to_numpy())
        if len(target_phases) == 2:
            stat = 'ttest'
            alternative = 'two-sided'
        else:
            stat = 'dft'
            alternative = 'greater'
        stat_fun = partial(_fooof_stat, 
                           measure=measure,  
                           freqs=df_data.attrs['freqs'], 
                           l_freq_target=df_data.attrs['l_freq_target'], 
                           h_freq_target=df_data.attrs['h_freq_target'],
                           stat=stat)
        res = permutation_test(target_psds,
                                lambda *x: stat_fun(x),
                                permutation_type='independent',
                                alternative=alternative,
                                n_resamples=1000)
        tval = res.statistic
        pval = res.pvalue
        tval_unit = stat
        if plot:
            plt.figure()
            x = target_phases
            x = [round(np.rad2deg(ph)) for ph in x]
            y = _fooof_agg(target_psds, 
                           measure=measure, 
                           freqs=df_data.attrs['freqs'], 
                           l_freq_target=df_data.attrs['l_freq_target'], 
                           h_freq_target=df_data.attrs['h_freq_target'])
            df_plot = pd.DataFrame(
                {'Target Phase (°)': x, '{}'.format(measure): y})
            sns.barplot(
                df_plot,
                x='Target Phase (°)',
                y='{}'.format(measure),
                color='k',
                alpha=0.5,
                errorbar=None,
                zorder=0)
            if len(target_phases) == 2:
                plt.title('t = {:.3e}, p = {:.3e}'.format(tval, pval))
            else:
                avgs = df_plot['{}'.format(measure)].to_numpy()
                _dft(avgs, plot_sine=True)
                plt.title('dft_amp = {:.3e}, p = {:.3e}'.format(tval, pval))
            sns.despine()
            if isinstance(plot, str):
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42
                sns.set_context('paper')
                if not os.path.exists(plot):
                    os.makedirs(plot,exist_ok=True)
                plt.savefig('{}_modulation_{}.pdf'.format(measure, participant))
                plt.close()
        df_append = pd.DataFrame({'participant': [participant],
                                  't_unit': [tval_unit],
                                  't_value': [tval],
                                  'p_value': [pval]})
        df_results = pd.concat([df_results, df_append])
    return df_results

def _test_modulation_psd_group(df_data, measure, plot):
    gb_target_phase = df_data.groupby('target_phase')
    target_phases = []
    target_psds = []
    for target_phase, df_target_phase in gb_target_phase:
        target_phases.append(target_phase)
        target_psds.append(df_target_phase['value'].to_numpy())
    if len(target_phases) == 2:
        stat = 'ttest'
        alternative = 'two-sided'
    else:
        stat = 'dft'
        alternative = 'greater'
    stat_fun = partial(_fooof_stat, 
                        measure=measure, 
                        freqs=df_data.attrs['freqs'], 
                        l_freq_target=df_data.attrs['l_freq_target'], 
                        h_freq_target=df_data.attrs['h_freq_target'],
                        stat=stat)
    res = permutation_test(target_psds,
                            lambda *x: stat_fun(x),
                            permutation_type='independent',
                            alternative=alternative,
                            n_resamples=1000)
    tval = res.statistic
    pval = res.pvalue
    tval_unit = stat
    if plot:
        plt.figure()
        x = target_phases
        x = [round(np.rad2deg(ph)) for ph in x]
        y = _fooof_agg(target_psds, 
                        measure=measure, 
                        freqs=df_data.attrs['freqs'], 
                        l_freq_target=df_data.attrs['l_freq_target'], 
                        h_freq_target=df_data.attrs['h_freq_target'])
        df_plot = pd.DataFrame(
            {'Target Phase (°)': x, '{}'.format(measure): y})
        sns.barplot(
            df_plot,
            x='Target Phase (°)',
            y='{}'.format(measure),
            color='k',
            alpha=0.5,
            errorbar=None,
            zorder=0)
        if len(target_phases) == 2:
            plt.title('t = {:.3e}, p = {:.3e}'.format(tval, pval))
        else:
            avgs = df_plot['{}'.format(measure)].to_numpy()
            _dft(avgs, plot_sine=True)
            plt.title('dft_amp = {:.3e}, p = {:.3e}'.format(tval, pval))
        sns.despine()
        if isinstance(plot, str):
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            sns.set_context('paper')
            if not os.path.exists(plot):
                os.makedirs(plot,exist_ok=True)
            plt.savefig('{}_modulation_group.pdf'.format(measure))
            plt.close()
    df_results = pd.DataFrame({'participant': ['group'],
                              't_unit': [tval_unit],
                              't_value': [tval],
                              'p_value': [pval]})
    return df_results

def get_network_modulation_amp_phase(df_network_results, df_network_data):
    
    """Get amplitude (depth) and optimal phase of modulation of connectivity
    in a network.

    Parameters:
    -----------
    df_network_results: pandas.DataFrame
        The dataframe containing the connections which were modulated at the 
        participant- or group-level.
        
    df_network_data: pandas.DataFrame
        The dataframe containing the connectivity matrix for each trial 
        and participant.
    """
    
    participants = np.unique(df_network_data['participant'])
    measure = df_network_data['measure'].iloc[0]
    df_results = pd.DataFrame()
    
    for participant in participants:
        df_network_data_participant = df_network_data[df_network_data['participant'] == participant]
        df_network_results_participant = df_network_results[df_network_results['participant'] == participant]
        
        network_data_participant, target_phases = df_to_array(df_network_data_participant)
        # this is now (n_phases, n_epochs, n_chs, n_chs)
        
        for ix_cluster, cluster in df_network_results_participant.iterrows():
            
            t_value = np.mean(cluster['t_values'])  # average over t-values per connection in cluster
            p_value = cluster['p_value']            # this is one p-value for the cluster
            ixs_row, ixs_col = np.array(cluster['connections']).T    # this contains the indices marking connections between sensors in cluster
            cluster_data = network_data_participant[:, :, ixs_row, ixs_col].mean(-1)
            avgs = cluster_data.mean(-1)
            amp, phase = _dft(avgs)
            df_append = pd.DataFrame({'participant':[participant],
                                      'measure':[measure],
                                      'cluster':[ix_cluster],
                                      'amplitude_raw':[amp],
                                      'amplitude_perc':[2*100*amp/np.mean(avgs)],
                                      'phase_rad':[phase],
                                      'phase_deg':[round(np.rad2deg(phase))]})
            df_results = pd.concat([df_results, df_append])
    return df_results