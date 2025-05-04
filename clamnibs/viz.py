import numpy as np
from mne.utils.check import _check_sphere
from mne.viz.topomap import _get_pos_outlines, _draw_outlines
from matplotlib import rcParams
import matplotlib.pyplot as plt
from functools import partial
from .base import RawCLAM, EpochsCLAM
from .misc import df_to_array
import seaborn as sns
from mne.viz import plot_sensors
import pandas as pd
from .stats import _dft
from mne import pick_info, pick_types

def _onpick_sensor(event, fig, ax, pos, ch_names, bads, scatter):
    if event is not None:
        if event.mouseevent.inaxes != ax:
            return
        ind = event.ind[0]
        ch_name = ch_names[ind]
        if ch_name in bads:
            bads.remove(ch_name)
        else:
            bads.append(ch_name)
            
    edgecolors = ['r' if ch in bads else 'k' for ch in ch_names]
    scatter.set_edgecolors(edgecolors)
    fig.canvas.draw()


def set_bads(obj, default_bads):
    
    """Interactive tool to mark bad channels in EEG data.

    This function provides an interactive visualization to mark bad channels in EEG data.
    It displays a scalp plot with the channels labeled, allowing the user to click on
    channels to mark them as bad.

    Parameters:
    -----------
    obj : RawCLAM or EpochsCLAM object
        The RawCLAM or EpochsCLAM object containing EEG data.
    default_bads: list of str
        The channels that are marked bad by default

    Raises:
    -------
    Exception:
        If the input object is not an instance of RawCLAM or EpochsCLAM.

    Notes:
    ------
    - This function requires Matplotlib to display the interactive visualization.
    - Bad channels are marked by clicking on the corresponding channel in the plot.
    - The plot window must be closed to finalize the selection of bad channels.
    - The updated list of bad channels is stored in the `bads` attribute of the input object.
    """
    
    if not (isinstance(obj, RawCLAM) or isinstance(obj, EpochsCLAM)):
        raise Exception('set_bads can only be applied to RawCLAM or EpochsCLAM objects')

    info = pick_info(obj.info, pick_types(obj.info, eeg=True, exclude=[]))
    ch_names = info.ch_names

    pos = np.empty((len(info["chs"]), 3))
    for ci, ch in enumerate(info['chs']):
        pos[ci] = ch["loc"][:3]

    sphere = _check_sphere(None, info)

    subplot_kw = dict()
    fig, ax = plt.subplots(1, figsize=(
        max(rcParams["figure.figsize"]),) * 2, subplot_kw=subplot_kw)

    ax.text(0, 0, "", zorder=1)

    pos, outlines = _get_pos_outlines(
        info, range(len(ch_names)), sphere, to_sphere=True)
    _draw_outlines(ax, outlines)
    # DRAW SERIES OF LINES HERE FOR CONNECTIVITY GRAPH
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        picker=True,
        clip_on=False,
        c='k',
        edgecolors='k',
        s=150,
        lw=2,
    )

    ax.set(aspect="equal")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.axis("off")

    indices = range(len(pos))
    for idx in indices:
        this_pos = pos[idx]
        ax.text(
            this_pos[0],
            this_pos[1] - 0.007,
            ch_names[idx],
            ha="center",
            va="center",
        )
    xmin, ymin, xmax, ymax = fig.get_window_extent().bounds
    renderer = fig.canvas.get_renderer()
    extents = [x.get_window_extent(renderer=renderer) for x in ax.texts]
    xmaxs = np.array([x.max[0] for x in extents])
    bad_xmax_ixs = np.nonzero(xmaxs > xmax)[0]
    if len(bad_xmax_ixs):
        needed_space = (xmaxs[bad_xmax_ixs] - xmax).max() / xmax
        fig.subplots_adjust(right=1 - 1.1 * needed_space)

    bads = [ch for ch in default_bads if ch in ch_names]
    
    picker = partial(
        _onpick_sensor,
        fig=fig,
        ax=ax,
        pos=pos,
        ch_names=ch_names,
        bads=bads,
        scatter=scatter
    )
    fig.canvas.mpl_connect("pick_event", picker)
    picker(None) # call to the update function with a dummy event to force it to draw the initial bad channels
    fig.set_size_inches(10, 12)
    ax.text(
        0.05,
        0.95,
        'Please mark all bad channels by clicking on them.\nClose the window when done',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='center',
        fontsize=14,
        weight='bold')
    plt.tight_layout()
    plt.show()
    obj.info['bads'] = bads
    
def plot_network_modulation_values(df_network_results, df_network_data, participant_identified_in, participant_applied_to):
    
    """Plot connectivity values averaged within each modulated network as a 
    box-/stripplot featuring each target phase.

    Parameters:
    -----------
    df_network_results: pandas.DataFrame
        The dataframe containing the connections which were modulated at the 
        participant- or group-level.
        
    df_network_data: pandas.DataFrame
        The dataframe containing the connectivity matrix for each trial 
        and participant.
        
    participant_identified_in : str
        The participant in which the network was identified. This network mask
        will be applied to the data and plotted. Can be 'group' for group-level 
        network.
        
    participant_applied_to : str
        The participant to whose data the network mask should be applied for
        plotting. Can be 'group' for group-level data.
    """
    
    df_network_results = df_network_results[df_network_results['participant'] == participant_identified_in]
    if participant_applied_to == 'group':
        df_network_data = df_network_data.groupby([col for col in df_network_data.columns if col != 'value']).agg({'value' : np.mean}).reset_index()
    else:
        df_network_data = df_network_data[df_network_data['participant'] == participant_applied_to]
    
    network_data, target_phases = df_to_array(df_network_data)
    # this is now (n_phases, n_epochs, n_chs, n_chs)
    
    measure = df_network_data['measure'].iloc[0]
    
    for ix_cluster, cluster in df_network_results.iterrows():
        
        t_value = np.mean(cluster['t_values'])  # average over t-values per connection in cluster
        p_value = cluster['p_value']            # this is one p-value for the cluster
        ixs_row, ixs_col = np.array(cluster['connections']).T    # this contains the indices marking connections between sensors in cluster
        cluster_data = network_data[:, :, ixs_row, ixs_col].mean(-1)
        x = np.concatenate([[target_phases[ix]] * len(cluster_data[ix])
                            for ix in range(len(target_phases))])
        x = [round(np.rad2deg(ph)) for ph in x]
        y = np.concatenate(cluster_data)
        df_plot = pd.DataFrame(
                {'Target Phase (°)': x, '{}'.format(measure): y})
        df_plot_agg = df_plot.sort_values('Target Phase (°)').groupby('Target Phase (°)') \
                .agg({'{}'.format(measure) : np.mean}).reset_index()
        plt.figure()
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
        if len(target_phases) > 2:
            avgs = df_plot_agg['{}'.format(measure)].to_numpy()
            _dft(avgs, plot_sine=True)
        plt.title('Identified in {}, applied to {}, Cluster {:d}, p = {:.3e}'.format(participant_identified_in, 
                                                                                participant_applied_to, 
                                                                                int(ix_cluster), 
                                                                                p_value))

def plot_network_modulation_topo(df_network_results, n_conns, participant, info):
    
    """Plot modulated network as connections between sensors on a topoplot.

    Parameters:
    -----------
    df_network_results: pandas.DataFrame
        The dataframe containing the connections which were modulated at the 
        participant- or group-level.
    
    n_conns : int
        The number of connections to plot. The n_conns most strongly modulated
        connections will be plotted.
        
    participant : str
        The participant in which the modulated network was identified. Can be 'group'
        for a group-level plot.
        
    info : mne.Info
        The Info object for topographic plotting.
    """
    
    df_network_results = df_network_results[df_network_results['participant'] == participant]
        
    colors = sns.color_palette("Set2")
    sphere = _check_sphere(None, info)
    pos, outlines = _get_pos_outlines(info, range(
        len(info.ch_names)), sphere, to_sphere=True)
    fig = plot_sensors(info, show_names=False, show=False)
    fig.set_size_inches(12, 12)
        
    for ix_cluster, cluster in df_network_results.iterrows():
        
        t_values = cluster['t_values']          # this is one t-value per connection in cluster
        p_value = cluster['p_value']            # this is one p-value for the cluster
        connections = cluster['connections']    # this contains the indices marking connections between sensors in cluster
        ixs_top_conns = np.argsort(t_values)[::-1][:n_conns]
        
        passed_label = False
        for ix_conn in ixs_top_conns:
            ix_ch_1, ix_ch_2 = connections[ix_conn]
            x_coords = [pos[ix_ch_1][0],
                        pos[ix_ch_2][0]]
            y_coords = [pos[ix_ch_1][1],
                        pos[ix_ch_2][1]]
            if not passed_label:
                label = 'Cluster {:d}'.format(int(ix_cluster))
                passed_label = True
            else:
                label = None
            plt.plot(x_coords, y_coords, c=colors[ix_cluster], label=label)
    plt.legend(frameon=False)
    plt.title('Identified in {}'.format(participant))
    plt.tight_layout()
    
def plot_modulation_amp_corr(df_amp_phase_results):
    
    """Plot correlation of amplitude (depth) of modulation of two 
    outcome measures cross participants.

    Parameters:
    -----------
    df_amp_phase_results: pandas.DataFrame
        The dataframe containing the amplitude (depth) of modulation
        for each participant and outcome measure.
    """
    
    # TODO
    # Do for all clusters in dataframe