import numpy as np
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
from scipy.sparse import coo_matrix
import scipy
from mne.stats import ttest_ind_no_p
from mne.viz.topomap import _get_pos_outlines
from mne.utils.check import _check_sphere
import matplotlib.pyplot as plt
import seaborn as sns
from mne.viz import plot_sensors
import mne
from mne.viz import plot_topomap
from functools import partial
import pandas as pd
from scipy.stats import ttest_ind
from numpy.random import permutation
from scipy.stats import permutation_test
import os
import matplotlib

# TODO:
# Statistics over sessions in session_wise studies

def _dft(x, plot_sine=False):
    n_bins = len(x)
    phases = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    c = (x*np.exp(-1j*phases)).sum()*2/n_bins
    amp, phase = np.abs(c), _wrap(np.angle(c))
    if plot_sine:
        n_bins = len(x)
        xs = np.linspace(-0.5, n_bins-0.5, 50)
        phases_xs = xs*2*np.pi/n_bins
        dy = np.mean(x)
        phases = np.linspace(phases_xs[0], phases_xs[-1],50)
        ys = amp*np.cos(phases + phase)+dy
        plt.plot(xs, ys, c='k', linewidth=3, zorder=2)
    return amp, phase

def _vectorized_dft_amp(*args):
    if args[0].ndim == 1:
        args = [arg[:, None] for arg in args]
    n_features = args[0].shape[-1]
    amps = np.empty(n_features)
    for ix_feature in range(n_features):
        x = np.array([np.mean(arg[:, ix_feature]) for arg in args])
        amp, _ = _dft(x)
        amps[ix_feature] = amp
    return amps

def _dft_amp_stat(*args):
    return _vectorized_dft_amp(*args)

# this performs permutations within each subject, hacky solution
def _dft_amp_stat_group(*args, orig_args=None):
    global first_pass_done
    if first_pass_done:
        permuted_args = np.array(orig_args).copy()
        for ix_participant in range(permuted_args.shape[1]):
            permuted_args[:, ix_participant] = permutation(
                permuted_args[:, ix_participant])
        return _vectorized_dft_amp(*permuted_args)
    else:
        first_pass_done = True
        return _vectorized_dft_amp(*orig_args)


def _wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi


def _fmt(string):
    return string.replace(
        '_',
        ' ').title().replace(
        ' Of ',
        ' of ').replace(
            ' And ',
        ' and ')


def test_sensor_network_modulation(
        df_data,
        info,
        test_level='participant',
        measure='phase_lag_index',
        plot=False):
    
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
        _test_sensor_network_modulation_participant(
            df_data, info, measure, plot)
    elif test_level == 'group':
        df_data = df_data.groupby([col for col in df_data.columns if col != 'value']).agg({'value' : np.mean}).reset_index()
        _test_sensor_network_modulation_group(df_data, info, measure, plot)
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_sensor_network_modulation_participant(df_data, info, measure, plot):
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
                [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_conns)], 95)
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
        if plot:
            colors = sns.color_palette("Set2")
            sphere = _check_sphere(None, info)
            pos, outlines = _get_pos_outlines(info, range(
                len(info.ch_names)), sphere, to_sphere=True)
            fig = plot_sensors(info, show_names=False, show=False)
            fig.set_size_inches(12, 12)
        tvals_sig = []
        pvals_sig = []
        conns_sig = []
        for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
            if pval < 0.05:
                tvals_sig.append(tvals[cluster].mean())
                pvals_sig.append(pval)
                conns_cluster = []
                for ix_conn in cluster[0]:
                    ix_ch_1 = ix_conn // n_chs
                    ix_ch_2 = ix_conn % n_chs
                    conns_cluster.append([ix_ch_1, ix_ch_2])
                    if plot:
                        x_coords = [pos[ix_ch_1][0],
                                    pos[ix_ch_2][0]]
                        y_coords = [pos[ix_ch_1][1],
                                    pos[ix_ch_2][1]]
                        plt.plot(x_coords, y_coords, c=colors[ix_cluster])
                conns_sig.append(conns_cluster)

        if plot:
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
                't_value': tvals_sig,
                'p_value': pvals_sig,
                'connections': conns_sig})
        df_results = pd.concat([df_results, df_append])
    return df_results


def _test_sensor_network_modulation_group(df_data, info, measure, plot):
    gb_target_phases = df_data.sort_values(
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
        threshold = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_obs - 1)
        tvals, clusters, pvals, _ = permutation_cluster_1samp_test(data[0] - data[1],
                                                                   threshold=threshold,
                                                                   adjacency=adjacency,
                                                                   out_type='indices',
                                                                   step_down_p=0,
                                                                   t_power=1,
                                                                   tail=0,
                                                                   n_jobs=1,
                                                                   verbose=True)
        tvals_unit = 'ttest_dep'
    else:
        stat_fun = _dft_amp_stat
        threshold = np.nanpercentile(
            [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_conns)], 95)
        global first_pass_done
        first_pass_done = False
        stat_fun = partial(
            _dft_amp_stat_group,
            orig_args=data)
        tvals, clusters, pvals, _ = permutation_cluster_test([d.copy() for d in data],
                                                             threshold=threshold,
                                                             adjacency=adjacency,
                                                             out_type='indices',
                                                             step_down_p=0,
                                                             t_power=1,
                                                             stat_fun=stat_fun,
                                                             tail=0,
                                                             n_jobs=1,
                                                             verbose=True,
                                                             buffer_size=n_conns)
        tvals_unit = 'dft_amp'
    if plot:
        colors = sns.color_palette("Set2")
        sphere = _check_sphere(None, info)
        pos, outlines = _get_pos_outlines(info, range(
            len(info.ch_names)), sphere, to_sphere=True)
        fig = plot_sensors(info, show_names=False, show=False)
        fig.set_size_inches(12, 12)
    tvals_sig = []
    pvals_sig = []
    conns_sig = []
    for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
        if pval < 0.05:
            tvals_sig.append(tvals[cluster].mean())
            pvals_sig.append(pval)
            conns_cluster = []
            for ix_conn in cluster[0]:
                ix_ch_1 = ix_conn // n_chs
                ix_ch_2 = ix_conn % n_chs
                conns_cluster.append([ix_ch_1, ix_ch_2])
                if plot:
                    x_coords = [pos[ix_ch_1][0],
                                pos[ix_ch_2][0]]
                    y_coords = [pos[ix_ch_1][1],
                                pos[ix_ch_2][1]]
                    plt.plot(x_coords, y_coords, c=colors[ix_cluster])
            conns_sig.append(conns_cluster)

    if plot:
        plt.title('Group, Modulation of {}'.format(_fmt(measure)))
        plt.tight_layout()
    df_results = pd.DataFrame({'participant': ['all'] * len(tvals_sig),
                              't_unit': [tvals_unit] * len(tvals_sig),
                              't_value': tvals_sig,
                              'p_value': pvals_sig,
                              'connections': conns_sig})
    return df_results


def test_sensor_cluster_modulation(
        df_data,
        info,
        test_level='participant',
        measure='amplitude',
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
        df_results = _test_sensor_cluster_modulation_participant(
            df_data, info, measure, plot)
        return df_results
    elif test_level == 'group':
        df_data = df_data.groupby([col for col in df_data.columns if col != 'value']).agg({'value' : np.mean}).reset_index()
        df_results = _test_sensor_cluster_modulation_group(df_data, info, measure, plot)
        return df_results
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_sensor_cluster_modulation_participant(df_data, info, measure, plot):
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
                [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_chs)], 95)
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
                tvals_sig.append(tvals[cluster].mean())
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
                't_value': tvals_sig,
                'p_value': pvals_sig,
                'channels': channels_sig})
        df_results = pd.concat([df_results, df_append])
    return df_results


def _test_sensor_cluster_modulation_group(df_data, info, measure, plot):
    gb_target_phases = df_data.sort_values(
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
        threshold = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_obs - 1)
        tvals, clusters, pvals, _ = permutation_cluster_1samp_test(data[0] - data[1],
                                                                   threshold=threshold,
                                                                   adjacency=adjacency,
                                                                   out_type='indices',
                                                                   step_down_p=0,
                                                                   t_power=1,
                                                                   tail=0,
                                                                   n_jobs=1,
                                                                   verbose=True)
        tvals_unit = 'ttest_dep'
    else:
        stat_fun = _dft_amp_stat
        threshold = np.nanpercentile(
            [stat_fun(*[d[:, ix] for d in data]) for ix in range(n_chs)], 95)
        global first_pass_done
        first_pass_done = False
        stat_fun = partial(
            _dft_amp_stat_group,
            orig_args=data)
        tvals, clusters, pvals, _ = permutation_cluster_test([d.copy() for d in data],
                                                             threshold=threshold,
                                                             adjacency=adjacency,
                                                             out_type='indices',
                                                             step_down_p=0,
                                                             t_power=1,
                                                             stat_fun=stat_fun,
                                                             tail=0,
                                                             n_jobs=1,
                                                             verbose=True,
                                                             buffer_size=n_chs)
        tvals_unit = 'dft_amp'
    tvals_sig = []
    pvals_sig = []
    channels_sig = []
    mask_sig = np.zeros(n_chs).astype(bool)
    for ix_cluster, (cluster, pval) in enumerate(zip(clusters, pvals)):
        if pval < 0.05:
            tvals_sig.append(tvals[cluster].mean())
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
    df_results = pd.DataFrame({'participant': ['all'] * len(tvals_sig),
                              't_unit': [tvals_unit] * len(tvals_sig),
                              't_value': tvals_sig,
                              'p_value': pvals_sig,
                              'channels': channels_sig})
    return df_results


def test_modulation(
        df_data,
        test_level='participant',
        measure='amplitude',
        agg_func=np.mean,
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
        
    agg_func : callable, optimal (default=np.mean)
        The function used to aggregate across trials within each phase bin (e.g. np.mean or np.std).
        
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
        _test_modulation_participant(df_data, measure, agg_func, plot, plot_mode)
    elif test_level == 'group':
        df_data = df_data.groupby([col for col in df_data.columns if col != 'value']).agg({'value' : agg_func}).reset_index()
        _test_modulation_group(df_data, measure, np.mean, plot, plot_mode)
    else:
        raise Exception(
            'Test level should be either \'participant\' or \'group\'')


def _test_modulation_participant(df_data, measure, agg_func, plot, plot_mode):
    df_results = pd.DataFrame()
    participants = df_data['participant'].unique()
    for participant in participants:
        df_participant = df_data[df_data['participant'] == participant]
        gb_target_phase = df_participant.groupby('target_phase')
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
    df_results = pd.DataFrame({'participant': ['all'],
                              't_unit': [tval_unit],
                              't_value': [tval],
                              'p_value': [pval]})
    return df_results