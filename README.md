# Go! Trail Ranger
## Project Structure
## 1. Unity  
Links to the the game specific data and assets in Unity Game Engine.  
## 2. Reinforcement Learning  
**AIBot.py -** Entry point where the AI bot learns the game using rewards and finally plays the game.  
**Agent.py -** Agent structure.  
**CNN.py -** Neural Networks brain that trains on the Game frames.  
**ExperienceReplay.py -** Experience Replay dealing with Reward training.  
**MovingAverage.py -** To calculate Moving average for the rewards.  
**ReplayMemory.py -** Stores the samples of the Replay memory.  
**SoftmaxBody.py -** Softmax function for the CNN classifier.  
**Utils.py -** Other Util functions.  
## 3. Imitation Learning
**Imitation_Learning_Train.ipynb -**  Train the model and generate the weights.  
**Classify_and_Play.ipynb -**  Use the weights to classify the game frames and play.  
## 4. Zero-shot Learning
**Zeroshot_Learning_Train.ipynb -**  Train the model and generate the weights.  
**Classify_and_Play.ipynb -**  Use the weights to classify the game frames and play.  
## 5. Transfer Learning
**Transfer_Learning_Train_Pass1.ipynb -**  Train the model using Subway Surfers game data and generate the initial weights.  
**Transfer_Learning_Train_Pass2.ipynb -**  Train the model using Go! Trail Ranger game data and generate the final weights.  
**Classify_and_Play.ipynb -**  Use the weights to classify the game frames and play.  
## 6. Transfer Learning Merge Network
**Parallel_Networks_Train.ipynb -**  Train the model and generate the weights.  
**Classify_and_Play.ipynb -**  Use the weights to classify the game frames and play.  
## 7. Results
**Graphs.ipynb -** Graphs.  
**Metrics.ipynb -**  Metrics.  
