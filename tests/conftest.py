import os
import sys
import pytest
import numpy as np

# Insert project root so clamnibs is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VHDR_PATH = os.path.join(DATA_DIR, 'task_stim.vhdr')

# Marker definitions for this dataset:
# S15 = trial start (simultaneous with S2 or S3 = condition codes)
# S16 = response (variable-latency end marker)
# S17/S18 = outcome
# S2/S3 appear simultaneously with S15 and serve as the phase condition codes
# For testing, we use S15 as start marker and S16 as end marker
# But the marker_definition maps trigger codes to phases, so we use S2 and S3
MARKER_DEFINITION = {
    2: 0.0,           # S2 -> phase 0
    3: np.pi,         # S3 -> phase pi
}

START_CODES = [15]    # S15 = trial onset
END_CODES = [16]      # S16 = response


@pytest.fixture(scope='session')
def raw_stim():
    """Load test RawCLAM object with stimulation data."""
    from clamnibs.base import RawCLAM
    raw = RawCLAM(
        VHDR_PATH,
        l_freq_target=8.0,
        h_freq_target=14.0,
        tmin=0,
        tmax=3,
        n_chs=64,
        design='trial_wise',
        ecg_channels=['ecg'],
        misc_channels=['envelope', 'envelope_am', 'eda'],
        marker_definition=MARKER_DEFINITION,
    )
    return raw


@pytest.fixture(scope='session')
def raw_stim_filtered(raw_stim):
    """RawCLAM filtered to target frequency range."""
    raw = raw_stim.copy().filter(8.0, 14.0)
    return raw


@pytest.fixture(scope='session')
def raw_stim_broadband(raw_stim):
    """RawCLAM filtered to broadband (1-30 Hz)."""
    raw = raw_stim.copy().filter(1.0, 30.0)
    return raw


@pytest.fixture(scope='session')
def epochs_fixed(raw_stim_filtered):
    """Fixed-length EpochsCLAM from filtered data."""
    from clamnibs.base import EpochsCLAM
    return EpochsCLAM(raw_stim_filtered)


@pytest.fixture(scope='session')
def epochs_variable(raw_stim_filtered):
    """Variable-length EpochsCLAMVariable from filtered data."""
    from clamnibs.base import EpochsCLAMVariable
    return EpochsCLAMVariable(
        raw_stim_filtered,
        end_codes=END_CODES,
        start_codes=START_CODES,
    )


@pytest.fixture(scope='session')
def epochs_variable_no_hil(raw_stim_filtered):
    """Variable-length EpochsCLAMVariable without Hilbert transform."""
    from clamnibs.base import EpochsCLAMVariable
    return EpochsCLAMVariable(
        raw_stim_filtered,
        end_codes=END_CODES,
        start_codes=START_CODES,
        apply_hil=False,
    )
