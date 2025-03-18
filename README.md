<p align="center"><img src="https://github.com/davidhaslacher/clam-nibs/assets/17557712/42767758-25fd-43ce-952e-53eda9240273" width="200"></p>

# CLAM-NIBS Toolbox

## Overview
This Python toolbox contains all software required to evaluate electroencephalography (EEG) or magnetoencephalography (MEG) data resulting from closed-loop amplitude-modulated non-invasive brain stimulation (CLAM-NIBS) experiments. A companion Matlab/Simulink toolbox to run these experiments will be released soon.

## Features
### Reconstruction of the target oscillation
Reconstruct the target oscillation in data recorded during stimulation using a beamformer.
### Extract single-trial metrics of brain activity and behavior
Extract amplitude/frequency of the target oscillation, large-scale sensor-space connectivity, or behavioral metrics on a single-trial basis.
### Phase-dependent modulation analysis
Analyze phase-dependent modulation of power or frequency of the target oscillation. Perform network-based permutation testing of phase-dependent modulation of large-scale connectivity.
### Support for various stimulation techniques
Support for EEG/MEG data recorded during amplitude-modulated transcranial alternating current stimulation (AM-tACS) [1], temporal interference stimulation (TIS) [2], and temporally interfering magnetic stimulation (TIMS) [3].
### Data-driven forward model computation
Use spatio-spectral decomposition [4] to identify a target oscillation and determine its forward model in data recorded in absence of stimulation.

## Installation
To install the CLAM-NIBS toolbox, simply run:

```sh
pip install clamnibs
```

## Usage
Here is an example of how to use the CLAM-NIBS toolbox to extract single-trial power spectral densities (PSDs) of activity in the target region and assess phase-dependent modulation of the target oscillation power:
```python
import mne
from clamnibs.base import RawCLAM
from clamnibs.source import compute_single_trial_psd
from clamnibs.stats import test_modulation_psd

# Load raw data
raw = RawCLAM('path/to/data.vhdr', 
        l_freq_target=8,
        h_freq_target=14,
        tmin=0,
        tmax=5,
        ecg_channels=['ecg'],
        misc_channels=['envelope',
                        'envelope_am',
                        'eda'],
        marker_definition={ 1: (0/6)*2*np.pi,
                            2: (1/6)*2*np.pi,
                            3: (2/6)*2*np.pi,
                            4: (3/6)*2*np.pi,
                            5: (4/6)*2*np.pi,
                            6: (5/6)*2*np.pi},
        sfreq=200
        )

# Filter data
raw.filter(1, 40)

# Compute single-trial PSDs
df_psd = compute_single_trial_psd(raw)

# Use permutation testing to assess phase-dependent modulation of power of the target oscillation
test_modulation_psd(df_psd, test_level='participant', measure='power', plot=True)
plt.show()
```

## Examples and Documentation

For more detailed examples and documentation, please refer to the following resources:

- [Jupyter notebooks with examples](examples/)
- [Documentation](clam-nibs.readthedocs.io)

## Which stimulation techniques does CLAM-NIBS encompass?
CLAM-NIBS denotes the closed-loop operation of all amplitude-modulated electric or magnetic brain stimulation techniques. This includes amplitude-modulated transcranial alternating current stimulation (AM-tACS) [1], temporal interference stimulation (TIS) [2], and temporally interfering magnetic stimulation (TIMS) [3].

## Why non-invasive brain stimulation?
Although invasive brain stimulation can be effective in treating neurological and psychiatric disorders, it is associated with medical risks, as well as costs associated with surgical implantation of stimulation and recording hardware. Non-invasive brain stimulation approaches do not carry with them the medical risks associated with surgery, and are often less expensive then invasive approaches. These methods can be more quickly implemented, and easily adjusted or discontinued.

## Why closed-loop?
Non-invasive brain stimulation is known to exhibit substantial effect variability. In part, this variability results from the dependency of stimulation effects on the state of the brain during stimulation. For instance, effects of transcranial alternating current stimulation (tACS) are known to depend on the phase of targeted brain oscillations. Closed-loop stimulation protocols adapt electric or magnetic stimulation the brain state in real-time, e.g. by precisely delivering stimulation at a certain phase difference relative to targeted brain oscillations.

## Why amplitude-modulated?
In order to adapt electric or magnetic stimulation to targeted brain oscillations in real-time, brain activity must be measured during stimulation. This is hampered by strong electromagnetic artifacts in the recorded electroencephalography (EEG) or magnetoencephalography (MEG) signal at the frequency of stimulation (i.e. the frequency of targeted brain oscillations). During amplitude-modulated stimulation, the artifact at the target frequency is substantially smaller than during conventional sinusoidal stimulation, enabling more effective assessment of targeted brain oscillations.

## References
[1] Haslacher, David, et al. "Working memory enhancement using real-time phase-tuned transcranial alternating current stimulation." Brain Stimulation (2024).<br>
[2] Violante, Ines R., et al. "Non-invasive temporal interference electrical stimulation of the human hippocampus." Nature neuroscience (2023).<br>
[3] https://patents.google.com/patent/US20240017083A1/en <br>
[4] Nikulin, Vadim V., et al. "A novel method for reliable and fast extraction of neuronal EEG/MEG oscillations on the basis of spatio-spectral decomposition." NeuroImage (2011).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for details on the code of conduct and the process for submitting pull requests.

## Contact
For any questions or issues, please open an issue on GitHub or contact the author at haslacherdavid@gmail.com.