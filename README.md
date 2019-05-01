# INFO8010: Deep learning

This repository is based on [BetaGo](https://github.com/maxpumperla/betago/). It is a environment dedicated to train deep learning model for game of go. The original environment is made for 19x19 games. All the files ending by "9x9" were created for the project to adapt the environment to 9x9 games. 

During this projet, we first train neural network from human move. In the second part of the work, we used  transfer learning from a pretrained model for the game of Go on a 19x19 board to a target model used on a 9x9 board.

Notice : All the training graph are stored in the  directory _graphic_.
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
To train a model from start to finish, we recommand to check end_to_end9x9.py file. It contains all the code needed to train a netork. It selects a processor in the processor_9x9.py file and a model in the model_9x9.py file. The dataset is contained in _tar.gz_ file in a directory data_9x9.

```{bash}
python end_to_end9x9.py
```

To become how to compare models + arguments for each program
# Filtering
The script used to filter games is in the file game_filtering.py. It could be used with the following command:
```{bash}
python game_filtering.py -s games_not_filtered -d games_filtered -f 2500
```

where games_not_filtered is a directory containing _.tar.gz_ archive. Each archive containing game recored in SGF format.
games_filtered is the destination directory and 2500 the threshold rank.
