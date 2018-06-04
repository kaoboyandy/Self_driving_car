# AI for Self Driving Car
## for autocpmplete: pip uninstall enum34
# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
# Getting the Q value

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__() 
        self.input_size = input_size # input size: number of dimensions (5 in our case) in the vector in your input state
        self.nb_action = nb_action # nb_action: number of possible action a car can go (right, left, straight)
        self.fc1 = nn.Linear(input_size, 50)  #from input layer to connecting/hidden layer
        self.fc2 = nn.Linear(50, 50)  #from input layer to connecting/hidden layer
        self.fc3 = nn.Linear(50, nb_action) #from hidden layer to output
       # nn.Linear(size of each input sample, size of each output sample, bias) applies a linear transformation to the incoming data y =ax + b 
    def forward(self, state):
        x = F.relu(self.fc1(state)) #rectifier function: function that has output = 0 for any input < 0 and output = input for any input >= 0
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x) 
        return q_values

# Implementing Experience Replay

class ReplayMemory(object): 
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event): #event has 4 states: state before (s-1), current state(s), action, reward
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size): #tool to standardize and get the sample. get some random sample from the memory. sample size to allow the network to learn from last 100 memory
        samples = zip(*random.sample(self.memory, batch_size))
        #"zip(*)" unzips the memory from (S1,a1, r1), (s2, a2, r2) to (s1, s2,s3), (a1, a2, a3), 
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
      # Variable converts them from tensors variable into pytorch variables with variable and pytorch
      # torch.cat concatenate according to the first dimension (basically stack them up on top of each other)
      # it is so that when we compute gradient descent, we will be able to update the rate



# Implementing Deep Q Learning
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma): 
        #create and initialise for our future learning.
        # input size: number of dimensions in the vector in your input state
        # nb_action: number of possible action a car can go (right, left, straight)
        # we could have capacity here but specify below instead
        # create an object model, memory, variable for last state & reward & action, optimizer,
        self.gamma = gamma
        self.reward_window = [] #mean of the evolving list of last 100 rewards, used to evaluate the evolution of the AI. We want to see this increasing over time
        self.model = Network(input_size, nb_action)  #create one neural network 
        self.memory = ReplayMemory(100000) #get 100000 sample transition and get 100 random samples to use
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # the ".parameters()" connects the adam optimizer to our neural network
        # optim contains all the tool to perform stochastic descent
        # optimize (learn) the weights of the neurons of our brain
        # choosing Adam optimizer which is a class. The argument connects the parameters of our neural network (Q value?) with learning rate
        # if learning rate too high then it won't learn properly with mistakes making. Trial and error
        self.last_state = torch.Tensor(input_size).unsqueeze(0) 
        # last state is a vector of 5 dimension (input size = 5) (left, straight, right, orientation, -orientation)        
        # for pytorch, it needs to be more than a vector, needs to be a torch tensor and also 1 more dimension corresponding to the batch. 
        # In batch because neural network cannot accept a single vector. Therefore we create a fake dimension corresponding to the batch.
        # unsqueeze(0) create the fake dimension (1st dimension)
        self.last_action = 0
        # action will be either  [0, 1, 2] which corresponds to the angle to rotation[0, 20, -20]
        self.last_reward = 0
    
    def select_action(self, state):
        #select the right action at each time
        #state = the input state of the network. The state has 5 dimensions (left, straight, right, orientation, -orientation)   
        # each state has 3 possible action
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # Temperature=100
        # generate a prob of the 3 possible state due to 3 diff possible actions that sums to 1.
        # Q function is a function of the state and action
        # softmax: attribute a higher prob to one with higher Q value 
        # State is a torch tensor that will beome self.last_state later. Convert torch tensor to torch variable and specify to exclude gradient
        # temperature parameter (0-100), the higher the value the more "sure" the AI is to choosing one with the higher the Q value 
           # softmax((1,2,3)) = [0.04,0.11,0.85] => softmax([1,2,3]*3) = [0,0.02,0.98]    
        action = probs.multinomial()
        # random draw of the action from the prob distribution made by "probs". multinomial distribution.
        # mulitnomial is the generalization of binomial distribution
        ### https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multinomial.html
        # multinomial logistic regression is where the dependent variable has more than 2 types of possible answers
        return action.data[0,0]
        # to get the action 0,1 or 2, it is contained in the data[0,0]. Due to the fake batch 
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): #transition of markov process
        #learn if exploration is going well
        # the "sample" definition has arrange our state in respect to time, arranged our transition in 4 forms and the "push" definition allows to use previous memory
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # get the action that resulted from the current state. "gather" allows us to take only the "action" of the matrix (1) instead of everything and because the action is at dimension 1, we need to select it first and then delete the fake dimension after extracting using squeeze(1)  
        # batch_action gives the action that was chosen for the state 
        # squeeze(1) to unbundle them from a batch into a vector or tensor. Batch only needed in neural network
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # the batch_next_state will be the a list of Q value for all possible state, detach all of them first then get the max among them. Action is in index. Then we use index "[0]" to go back to the "state"
        # "detach" all the transition state in "batch_next_state" so that you can take the max of it
        # max of the q value of the next state represented by all the action that is represented by index 1.
        # [0] goes to the Q value of "next State" using the acion
        #see the formulas in step 5 p.7 of the doc 
        target = self.gamma*next_outputs + batch_reward
        #see the formulas in step 5 p.7 of the doc 
        td_loss = F.smooth_l1_loss(outputs, target)
        # huber loss function using "smooth_l1_loss(prediction: the output of neural network is what NN predicts, the thing we're trying to get)
        self.optimizer.zero_grad()
        # reinitialize the optimizer at each duration of the loop of stochastic descent. 
        td_loss.backward(retain_variables = True)
        # backward propagation
        # "retain_variables = True" improve back propagation. It will freeze the memory to go several time on the loss improve the training.
        self.optimizer.step()
        # updates the weight using the optimizer from the backward propogation
    
    def update(self, reward, new_signal):
        # linked to "action = brain.update(last_reward, last_signal)" in the map.py
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # convert the self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation into float and create fake dimension using unsqueeze
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # updating the new transition to the memory. remember "event" has 4 variables. LongTensor converts the last_action (which is 0,1,2) into tensor. 
        #self.last_reward doesn't need int conversion since it's a float. 
        action = self.select_action(new_state)
        #play the action based on new_state
        if len(self.memory.memory) > 100: #learn from the last 100 events (if the number is greater than 100)
            # first memory is the self.memory refering to replaymemory then address the second memory by another memory
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # the 4 states in the memory gathered from 100 samples
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            #learning from all the batches
        
        ### the new becomes last
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
        #so that the "action = brain.update(last_reward, last_signal)" in map.py can act
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) 
    #add +1 to make sure the denominator is never equal to 0
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    #save a key:value dictionary. .state_dict() save the parameter of the model and optimizer.
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            #update the model and optimizer with the  weight that was saved in last_brain.pth file
            self.model.load_state_dict(checkpoint['state_dict']) #checkpoint variable 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")