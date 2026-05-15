"""Tests for variable-length epochs in CLAMNIBS."""
import numpy as np
import pytest

# ============================================================
# EpochsCLAMVariable creation and basic properties
# ============================================================

class TestEpochsCLAMVariableCreation:
    
    def test_creates_epochs(self, epochs_variable):
        assert len(epochs_variable) > 0
    
    def test_epochs_have_variable_length(self, epochs_variable):
        """Epochs should have different numbers of timepoints."""
        lengths = [ep.shape[1] for ep in epochs_variable._data]
        assert len(set(lengths)) > 1, \
            f'All epochs have the same length ({lengths[0]}), expected variable lengths'
    
    def test_events_shape(self, epochs_variable):
        assert epochs_variable.events.ndim == 2
        assert epochs_variable.events.shape[1] == 3
        assert epochs_variable.events.shape[0] == len(epochs_variable)
    
    def test_events_codes_are_condition_codes(self, epochs_variable):
        """Event codes should match marker_definition codes when available."""
        unique_codes = set(epochs_variable.events[:, 2])
        assert unique_codes == {2, 3}
    
    def test_durations(self, epochs_variable):
        assert len(epochs_variable.durations) == len(epochs_variable)
        for d in epochs_variable.durations:
            assert d > 0
    
    def test_durations_are_variable(self, epochs_variable):
        durations = epochs_variable.durations
        assert len(set(durations)) > 1
    
    def test_attributes_copied(self, epochs_variable, raw_stim_filtered):
        assert epochs_variable.participant == raw_stim_filtered.participant
        assert epochs_variable.session == raw_stim_filtered.session
        assert epochs_variable.design == raw_stim_filtered.design
        assert epochs_variable.l_freq_target == raw_stim_filtered.l_freq_target
        assert epochs_variable.h_freq_target == raw_stim_filtered.h_freq_target
        assert epochs_variable.is_stim == raw_stim_filtered.is_stim
        assert epochs_variable.n_chs == raw_stim_filtered.n_chs
        np.testing.assert_array_equal(
            epochs_variable.forward_full, raw_stim_filtered.forward_full)
        assert epochs_variable.flip == raw_stim_filtered.flip


# ============================================================
# get_data
# ============================================================

class TestGetData:
    
    def test_get_data_returns_list(self, epochs_variable):
        data = epochs_variable.get_data()
        assert isinstance(data, list)
        assert len(data) == len(epochs_variable)
    
    def test_get_data_shapes(self, epochs_variable):
        data = epochs_variable.get_data()
        n_chs_total = len(epochs_variable.ch_names)
        for ep in data:
            assert ep.ndim == 2
            assert ep.shape[0] == n_chs_total
    
    def test_get_data_picks_indices(self, epochs_variable):
        picks = [0, 1, 2]
        data = epochs_variable.get_data(picks=picks)
        for ep in data:
            assert ep.shape[0] == 3
    
    def test_get_data_picks_eeg(self, epochs_variable):
        data = epochs_variable.get_data(picks='eeg')
        for ep in data:
            # Should have 64 - n_bads EEG channels
            assert ep.shape[0] <= 64
            assert ep.shape[0] > 0
    
    def test_get_data_returns_copies(self, epochs_variable):
        data = epochs_variable.get_data()
        data[0][:] = 0
        original = epochs_variable._data[0]
        assert not np.all(original == 0)
    
    def test_hilbert_produces_complex(self, epochs_variable):
        """With apply_hil=True (default), data should be complex."""
        data = epochs_variable.get_data(picks='eeg')
        assert np.iscomplexobj(data[0])
    
    def test_no_hilbert_produces_real(self, epochs_variable_no_hil):
        data = epochs_variable_no_hil.get_data(picks='eeg')
        assert np.isrealobj(data[0])


# ============================================================
# Indexing
# ============================================================

class TestIndexing:
    
    def test_int_index(self, epochs_variable):
        ep = epochs_variable[0]
        assert isinstance(ep, np.ndarray)
        assert ep.ndim == 2
    
    def test_slice_index(self, epochs_variable):
        sliced = epochs_variable[1:3]
        from clamnibs.base import EpochsCLAMVariable
        assert isinstance(sliced, EpochsCLAMVariable)
        assert len(sliced) == 2
    
    def test_bool_index(self, epochs_variable):
        mask = np.zeros(len(epochs_variable), dtype=bool)
        mask[0] = True
        mask[2] = True
        sliced = epochs_variable[mask]
        from clamnibs.base import EpochsCLAMVariable
        assert isinstance(sliced, EpochsCLAMVariable)
        assert len(sliced) == 2


# ============================================================
# Beamformer: get_target with variable epochs
# ============================================================

class TestGetTargetVariable:
    
    def test_returns_list(self, epochs_variable):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_variable)
        assert isinstance(target, list)
        assert len(target) == len(epochs_variable)
    
    def test_each_element_is_1d(self, epochs_variable):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_variable)
        for t in target:
            assert t.ndim == 1
    
    def test_lengths_match_epochs(self, epochs_variable):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_variable)
        for t, ep in zip(target, epochs_variable._data):
            assert t.shape[0] == ep.shape[1]
    
    def test_sensible_values(self, epochs_variable):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_variable)
        for t in target:
            assert np.all(np.isfinite(t))
            assert not np.all(t == 0)


# ============================================================
# get_target with fixed epochs (backward compatibility)
# ============================================================

class TestGetTargetFixed:
    
    def test_returns_ndarray(self, epochs_fixed):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_fixed)
        assert isinstance(target, np.ndarray)
    
    def test_shape(self, epochs_fixed):
        from clamnibs.beamformer import get_target
        target = get_target(epochs_fixed)
        n_epochs = len(epochs_fixed)
        assert target.shape[0] == n_epochs


# ============================================================
# Analysis functions with variable-length epochs
# ============================================================

class TestAmplitudeVariable:
    
    def test_compute_amplitude(self, raw_stim_filtered):
        from clamnibs.source import compute_single_trial_amplitude
        df = compute_single_trial_amplitude(
            raw_stim_filtered, end_codes=[16])
        assert len(df) > 0
        assert 'value' in df.columns
        assert 'target_phase' in df.columns
        assert 'participant' in df.columns
        for v in df['value']:
            assert np.isfinite(v)
            assert v >= 0


class TestAmplitudeFixed:
    
    def test_compute_amplitude_backward_compat(self, raw_stim_filtered):
        from clamnibs.source import compute_single_trial_amplitude
        df = compute_single_trial_amplitude(raw_stim_filtered)
        assert len(df) > 0
        assert 'value' in df.columns


class TestPSDVariable:
    
    def test_compute_psd(self, raw_stim_broadband):
        from clamnibs.source import compute_single_trial_psd
        df = compute_single_trial_psd(
            raw_stim_broadband, end_codes=[16])
        assert len(df) > 0
        assert 'value' in df.columns
        for v in df['value']:
            assert np.all(np.isfinite(v))


class TestPSDFixed:
    
    def test_compute_psd_backward_compat(self, raw_stim_broadband):
        from clamnibs.source import compute_single_trial_psd
        df = compute_single_trial_psd(raw_stim_broadband)
        assert len(df) > 0


# ============================================================
# EpochsCLAM backward compatibility
# ============================================================

class TestEpochsCLAMBackwardCompat:
    
    def test_fixed_epochs_creation(self, epochs_fixed):
        assert len(epochs_fixed) > 0
    
    def test_fixed_epochs_data_shape(self, epochs_fixed):
        data = epochs_fixed.get_data()
        assert data.ndim == 3  # (n_epochs, n_channels, n_timepoints)
    
    def test_fixed_epochs_uniform_length(self, epochs_fixed):
        data = epochs_fixed.get_data()
        assert data.shape[2] > 0  # all same length by construction

    def test_tmin_tmax_required_for_fixed(self, raw_stim):
        """EpochsCLAM should fail if tmin/tmax are None."""
        from clamnibs.base import EpochsCLAM, RawCLAM
        import mne
        # Can't easily set tmin=None on existing raw, but we can test the check
        raw_copy = raw_stim.copy()
        raw_copy.tmin = None
        with pytest.raises(Exception, match='tmin and tmax must be set'):
            EpochsCLAM(raw_copy)


# ============================================================
# RawCLAM without tmin/tmax (for variable-length use)
# ============================================================

class TestRawCLAMNoTminTmax:
    
    def test_raw_creation_without_tmin_tmax(self):
        """RawCLAM should be creatable without tmin/tmax for variable-length use."""
        from clamnibs.base import RawCLAM
        raw = RawCLAM(
            VHDR_PATH,
            l_freq_target=8.0,
            h_freq_target=14.0,
            tmin=None,
            tmax=None,
            n_chs=64,
            design='trial_wise',
            ecg_channels=['ecg'],
            misc_channels=['envelope', 'envelope_am', 'eda'],
            marker_definition={2: 0.0, 3: np.pi},
        )
        assert raw.tmin is None
        assert raw.tmax is None


from conftest import VHDR_PATH
