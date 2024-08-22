from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='clamnibs',
    version='0.1.9',
    description='Analysis of closed-loop amplitude-modulated non-invasive brain stimulation experiments',
    long_description='This toolbox contains all software required to implement closed-loop amplitude-modulated non-invasive brain stimulation (CLAM-NIBS) and evaluate the resulting electroencephalography (EEG) or magnetoencephalography (MEG) data.',
    author='David Haslacher',
    author_email='haslacherdavid@gmail.com',
    url='https://github.com/davidhaslacher/clam-nibs',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)