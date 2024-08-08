<p align="center"><img src="https://github.com/davidhaslacher/clam-nibs/assets/17557712/42767758-25fd-43ce-952e-53eda9240273" width="200"></p>

## Overview
This toolbox contains all software required to implement closed-loop amplitude-modulated non-invasive brain stimulation (CLAM-NIBS) and evaluate the resulting electroencephalography (EEG) or magnetoencephalography (MEG) data.

## Which stimulation techniques does CLAM-NIBS encompass?
CLAM-NIBS denotes the closed-loop operation of all amplitude-modulated electric or magnetic brain stimulation techniques. This includes amplitude-modulated transcranial alternating current stimulation (AM-tACS) [1], temporal interference stimulation (TIS) [2], and temporally interfering magnetic stimulation (TIMS) [3].

## Why non-invasive brain stimulation?
Although invasive brain stimulation can be effective in treating neurological and psychiatric disorders, it is associated with medical risks, as well as costs associated with surgical implantation of stimulation and recording hardware. Non-invasive brain stimulation approaches do not carry with them the medical risks associatd with surgery, and are often less expensive then invasive approaches. These methods can be more quickly implemented, and easily adjusted or discontinued.

## Why closed-loop?
Non-invasive brain stimulation is known to exhibit substantial effect variability. In part, this variability results from the dependency of stimulation effects on the state of the brain during stimulation. For instance, effects of transcranial alternating current stimulation (tACS) are known to depend on the phase of targeted brain oscillations. Closed-loop stimulation protocols adapt electric or magnetic stimulation the brain state in real-time, e.g. by precisely delivering stimulation at a certain phase difference relative to targeted brain oscillations.

## Why amplitude-modulated?
In order to adapt electric or magnetic stimulation to targeted brain oscillations in real-time, brain activity must be measured during stimulation. This is hampered by strong electromagnetic artifacts in the recorded electroencephalography (EEG) or magnetoencephalography (MEG) signal at the frequency of stimulation (i.e. the frequency of targeted brain oscillations). During amplitude-modulated stimulation, the artifact at the target frequency is substantially smaller than during conventional sinusoidal stimulation, enabling more effective assessment of targeted brain oscillations.

## References
[1] Haslacher, David, et al. "Working memory enhancement using real-time phase-tuned transcranial alternating current stimulation." Brain Stimulation (2024).
[2] Violante, Ines R., et al. "Non-invasive temporal interference electrical stimulation of the human hippocampus." Nature neuroscience (2023).
[3] https://patents.google.com/patent/US20240017083A1/en