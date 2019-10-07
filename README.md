# Reinforcement learning agents for StarCraft II micromanagement

A tabular version implementation of Sarsa(lambda) that is used to play various micromanagement scenarios. The maps for 
these scenarios are given in the sarsa maps directory.

## Install
Install dependencies while in the root directory:
`pipenv install`

Make sure you have pipenv installed. Else run ```pip install pipenv```

## Run
To run the sarsa agent (version 2 is updated version):
`python sarsa_agent_v2.py`

The above command accepts command line arguments:

| Argument flag | Description |
| --- | --- |
| `--map` | map name |
| `--num_of_eps` | number of game instances to run training |
| `--normalize` | apply normalisation to reward |
| `--advantage` | apply advantage operator to q table |
| `--test_id` | name for test run and result file prefix |
| `--train` | to launch fresh training |
| `--aggression` | aggression value to use for reward function |


## Results
https://i.cs.hku.hk/fyp/2018/fyp18045/
