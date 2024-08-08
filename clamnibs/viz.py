import numpy as np
from mne.utils.check import _check_sphere
from mne.viz.topomap import _get_pos_outlines, _draw_outlines
from matplotlib import rcParams
import matplotlib.pyplot as plt
from functools import partial
from .base import RawCLAM, EpochsCLAM

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

    info = obj.info
    ch_names = obj.ch_names

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