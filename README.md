# AI-HS-gameplay

AI Model to play the board game "High Society". Training the agent using Deep Q-learning. We train in two phases for now. First, we train against three random opponents - players we make moves completely randomly. Next, we train against earlier versions of itself in the training process. 

To train agent for 100 iterations, run:

python main.py train --num_players 4 --episodes 100

Note that this initialises both phases of training. 

To evaluate against opponents that make random moves, run: 

python main.py eval --num_players 4 --episodes 100

To evaluate against earlier versions of itself, run: 

python script.py eval --strategic --num_players 4 --episodes 100 