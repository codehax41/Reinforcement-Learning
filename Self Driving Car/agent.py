import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Create the Architecture
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #fully connected layer: means each i/p neauron will be connected to hidden layer
        #30 neuron in Hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    #foreward propagation
    #it will return q value for each possible action
    def foreward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Experience Replay
#Store 100 examples
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] #memory
    #v7
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            #remove first eleemnt to make sure we have only required episodes
            del self.memory[0]
    
    def sample(self, batch_size):
        #taking random sample from memory of batch size
        samples = zip(*random.sample(self.memory, batch_size))
        #return torch var, each batch is a pytorch variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

#Deep Q Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(10000) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        self.last_state = torch.Tensor(input_size).unsqueeze(0) 
        self.last_action = 0 
        self.last_reward = 0 
        
    def select_action(self, state): 
        probe = F.softmax(self.model(Variable(state, volatile=True))*100) #7
        action = probs.multinomial() 
        return action.data[0,0] 

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): 
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) 
        next_outputs = self.model(batch_next_state).detach().max(1)[0] 
        target = self.gamma*next_outputs + batch_reward 
        td_loss = F.smooth_l1_loss(outputs, target) 
        self.optimizer.zero_grad() 
        td_loss.backward(retain_variables = True) 
        self.optimizer.step()
    #so when ai reaches new state, old state become new, last action become new action, last reawrd->new
    #so we need to update all the transition to get new,
    #by giving last reward and last signal it will give new action based on updated values
    #we will update the action function which is select_action, so we will integrate select action function,
    #in the future update fn (below one) to select right action to play beside making all the updates 
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            #so it will learn from 100 samples from memory
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window)>1000:
            del self.reward_window[0]
        return action
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
        'optimizer':self.optimizer.state_dict,
        }, 'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            print('done!')
        else:
            print('no checkpoint found!...')