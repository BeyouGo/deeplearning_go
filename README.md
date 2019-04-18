# INFO8010: Deep learning

This repository contains an attempt of transfer learning from a pretrained model for the game of Go on a 19x19 board to a target model used on a 9x9 board.

## How to use
### Installation
First set up the prerequisites:

```{bash}
sudo apt-get install -y python-dev python-pip python-virtualenv gfortran libhdf5-dev pkg-config liblapack-dev libblas-dev
```

Then create and setup the environment:

```{bash}
virtualenv .betago
. .betago/bin/activate
pip install --upgrade pip setuptools
git clone https://github.com/BeyouGo/deeplearning_go.git
cd betago
python end_to_end9x9.py
```

### Execution
To train a model from scratch using a pretrained model (19x19) as core.

```{bash}
python end_to_end9x9.py
```

To become how to compare models + arguments for each program

## Current tasks
### Research
* [ ] Find how to compare AIs
* [ ] Research what types of model to use or alteration of models

### Code
* [ ] Check when game is over and actually end the game when playing
* [ ] Save models once trained
* [ ] Automate the AI comparison using saved models and weights
* [ ] Document how to use

### Execution
* [ ] Train models

### Reporting
* [ ] Summarize work done
* [ ] Summarize results obtained
* [ ] Slides
