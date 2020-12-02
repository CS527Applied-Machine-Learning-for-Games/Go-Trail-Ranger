# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import time

import Agent
import CNN
import SoftmaxBody as Body
import MovingAverage as Avg
import ExperienceReplay as Er
import ReplayMemory as Rm


# Save the Model and Optimizer's state (weights, bias, and parameters) to a file
def save_brain():
    torch.save({'model_state_dict': cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'recent_brain.pth')


# Load the Model and Optimizer's state (weights, bias, and parameters) from a recently saved file if present
def load_brain():
    if os.path.isfile('recent_brain.pth'):
        print("=> Loading recently saved brain ...")
        checkpoint = torch.load('recent_brain.pth')
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> Loaded !")
    else:
        print("=> No recently saved brain found !")


# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output, hidden = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


# Training the AI Bot with DQN Deep Convolutional Q-Learning
# Building an AI
cnn = CNN.CNN(number_actions=5)  # Up, Down, Left, Right, No movement
cnn = cnn.to("cpu")

softmax_body = Body.SoftmaxBody(T=1.0)
agent = Agent.Agent(brain=cnn, body=softmax_body)

# Setting up Experience Replay
n_steps = Er.NStepProgress(ai=agent, n_step=5)
memory = Rm.ReplayMemory(n_steps=n_steps, capacity=1000)

# Training the AI
ma = Avg.MovingAverage(100)
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0007)

nb_epochs = 400

load_brain()           # Uncomment to load previously trained Brain

for epoch in range(1, nb_epochs + 1):
    memory.run_steps(100)  # One epoch consists of 100 steps
    print("=> Entering epoch " + str(epoch))

    for batch in memory.sample_batch(72):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions, hidden = cnn(inputs, None)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()

    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

    save_brain()
    time.sleep(4)

    if avg_reward >= 200:
        print("Congratulations, your AI wins")
        save_brain()
        break
