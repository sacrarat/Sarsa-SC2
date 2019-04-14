import numpy as np
import pandas as pd

# some inspiration from https://github.com/skjb/pysc2-tutorial
# adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.5, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay)

        # state dictionary to keep track of the number of times the state is visited.
        self.state_counter = {}
        self.epsilon = e_greedy
        self.train = 1

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
            self.state_counter[state] = [1, self.epsilon]
        else:
            # get the current counter and epsilon value
            current_counter = self.state_counter[state][0]
            current_epsilon = self.state_counter[state][1]

            # increase epsilon linearly from 0.5 to 0.95 over 500 visits
            if current_counter < 500:
                # increase the counter of state
                updated_counter = current_counter + 1
                updated_epsilon = (updated_counter / 500) * 0.45 + self.epsilon
                self.state_counter[state] = [updated_counter, updated_epsilon]

    def apply_advantage(self):
        # update q table, loop through each state
        col = list(self.q_table)
        for i, row in self.q_table.iterrows():
            mean = row.mean()
            for j in col:
                self.q_table.at[i, j] -= mean

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if self.train == 0:
            use_epsilon = 1
        else:
            use_epsilon = self.state_counter[observation][1]
        # action selection
        if np.random.rand() < use_epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        self.eligibility_trace.loc[s, a] += 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_
