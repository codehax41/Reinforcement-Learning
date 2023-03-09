import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):

        self.state = [np.nan for _ in range(9)]
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] 
        self.reset()

    def reset(self):
        return self.state

    def is_winning(self, curr_state):
        win_sum = 15
        if ((curr_state[0] + curr_state[1] + curr_state[2]) ==  win_sum) or \
           ((curr_state[3] + curr_state[4] + curr_state[5]) == win_sum) or \
           ((curr_state[6] + curr_state[7] + curr_state[8]) == win_sum) or \
           ((curr_state[0] + curr_state[3] + curr_state[6]) == win_sum) or \
           ((curr_state[1] + curr_state[4] + curr_state[7]) == win_sum) or \
           ((curr_state[2] + curr_state[5] + curr_state[8]) == win_sum) or \
           ((curr_state[0] + curr_state[4] + curr_state[8]) == win_sum) or \
           ((curr_state[2] + curr_state[4] + curr_state[6]) == win_sum):
            return True
        else:
            return False

    def is_terminal(self, curr_state):
        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'
        else:
            return False, 'Resume'

    def allowed_positions(self, curr_state):
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)

    def action_space(self, curr_state):
        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)
      
    def state_transition(self, curr_state, curr_action):
        resultant_state = curr_state
        resultant_state[curr_action[0]] = curr_action[1]
        return resultant_state

    def reward(self, status):
        if status == 'Win':
            return 10
        elif status == 'Tie':
            return 0
        elif status == 'Resume':
            return -1
        else:
            return -10

    def step(self, curr_state, curr_action):
        agent_actions, env_actions = self.action_space(curr_state)
        assert len([action for action in agent_actions if action == curr_action]) >= 1, 'invalid action'

        resultant_state = [state for state in curr_state]
        resresultant_state = self.state_transition(resultant_state, curr_action)
        is_terminated, status = self.is_terminal(resultant_state)
        reward = self.reward(status)

        if is_terminated == False:
            agent_actions, env_actions = self.action_space(resultant_state)
            env_action = random.choice([action for action in env_actions])
            resultant_state = self.state_transition(resultant_state, env_action)
            is_terminated, status = self.is_terminal(resultant_state)
            if (is_terminated == True) and (status == 'Win'):
                reward = self.reward('Loss')
        return (resultant_state, reward, is_terminated)


