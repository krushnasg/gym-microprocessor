# gym-microprocessor
A simple OpenAI environment for training RL agents on microprocessor simulations.

## Step #1: Setup your python environment
(skip this step if you have a virtual environment ready already)
#### Install virtualenv and virtualenvwrapper
```
$ sudo pip install virtualenv virtualenvwrapper
$ sudo rm -rf ~/get-pip.py ~/.cache/pip
```
Once we have virtualenv and virtualenvwrapper installed, we need to update our ```~/.bashrc```
file to include the following lines at the bottom of the file:
```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```
After editing our ```~/.bashrc```   file, we need to reload the changes:
```
$ source ~/.bashrc
```
***Note**: Calling ```source``` on ```.bashrc``` only has to be done once for our current shell session. 
Anytime we open up a new terminal, the contents of ```.bashrc``` will be automatically executed (including our updates).*

Now that we have installed virtualenv and virtualenvwrapper, the next step is to actually create the Python virtual environment — we do this using the mkvirtualenv command.

#### Creating your Python virtual environment
```
$ mkvirtualenv rl -p python3
```
Regardless of which Python command you decide to use, the end result is that we have created a Python virtual environment named *rl* (short for Reinforcement Learning)

## Step #2: Installing the requirements 
Now that we have the virtual environment ready, we activate it using:
```
$ workon rl
```

All the required dependencies for running this project are present in the file ```requirements.txt```.
The dependencies can be installed using:
```
$ pip install -r requirements.txt
```
## Step #3: Understanding the workflow of the project
The directory structure of the project is as follows:
```
.
├── data
│   └── example.xlsx
├── gym_microprocessor
│   ├── envs
│   │   ├── __init__.py
│   │   ├── microprocessor_env.py
│   │   └── processor.py
│   │   
│   └── __init__.py
├── main.py
├── out.csv
├── README.md
├── requirements.txt
└── setup.py
```
The training algorithm is driven by file ```main.py```.
The directory ```gym_microprocessor``` contains the folder ```envs```, which has two important files:
1. ```processor.py``` contains the classes for Processor, Core, and Task. These essentially define the working of each of these entities.
2. ``` microprocessor_env.py``` contains the ```gym``` environment that is used for training the RL algorithm. 
The major driving functions for the environment are ```reset```, ```step```, and ```render```

