from q_network import DQLAgent, ReplayBuffer, Q_network, State
from play import play_game, play_strategic_game, play_game_eval, play_strategic_game_eval
import torch
import os 



NUM_STATES = 557 # state vector size 
NUM_ACTIONS = 2048 # Number of possible combinations of bids 

# Define the game environment and necessary parameters
state_size = NUM_STATES
action_size = NUM_ACTIONS
hidden_size = 1024
learning_rate = 0.0001 
gamma = 0.99
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.01  # Minimum exploration rate
epsilon_decay = 0.001  # Decay rate of epsilon over time
buffer_size = 10000   # maximium size of replay buffer
batch_size = 32    # size of each minibach 
min_epsilon = 0.01 
seed = 123

NUM_STATES = 546 # state vector size 
NUM_ACTIONS = 2048 # Number of possible combinations of bids 

NUMBERED_CARDS = [f'{i}' for i in range(1,11)]
SPECIAL_POSITIVE_CARDS = ['x2', 'x2', 'x2']
SPECIAL_NEGATIVE_CARDS = ['1/2', '-5', 'discard']



# def train(training_episodes: int, num_players: int = 4):
#     weights_dir = 'Weights'
#     if not os.path.exists(weights_dir):
#         os.makedirs(weights_dir)

#     agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
#     weights_path = os.path.join(weights_dir, f'weights_{training_episodes}.pth')
#     torch.save(agent.q_network.state_dict(), weights_path)

#     for _ in range(training_episodes):
#         agent.q_network.load_state_dict(torch.load(weights_path))
#         play_game(agent, num_players)
#         torch.save(agent.q_network.state_dict(), weights_path)


def eval_against_random_opponents(training_episodes:int=100, num_players:int=4):
    '''
    Play a game against 3 opponents that make random moves     
    '''
    agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    weights_dir = 'Weights'
    weights_path = os.path.join(weights_dir, f'weights_{training_episodes}_strategic.pth')
    agent.q_network.load_state_dict(torch.load(weights_path))
    play_game_eval(agent, num_players)


def eval_against_strategic_opponents(training_episodes:int=100, num_players:int=4):
    '''
    Play a game against 3 opponents that make random moves     
    '''
    agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    weights_dir = 'Weights'
    weights_path = os.path.join(weights_dir, f'weights_{training_episodes}_strategic.pth')
    agent.q_network.load_state_dict(torch.load(weights_path))

    opponent_1 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    weights_dir = 'Weights'
    weights_path = os.path.join(weights_dir, f'weights_{training_episodes - 1}_strategic.pth')
    opponent_1.q_network.load_state_dict(torch.load(weights_path))

    opponent_2 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    weights_dir = 'Weights'
    weights_path = os.path.join(weights_dir, f'weights_{training_episodes - 2}_strategic.pth')
    opponent_2.q_network.load_state_dict(torch.load(weights_path))

    opponent_3 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    weights_dir = 'Weights'
    weights_path = os.path.join(weights_dir, f'weights_{training_episodes - 3}_strategic.pth')
    opponent_3.q_network.load_state_dict(torch.load(weights_path))
    
    play_strategic_game_eval(agent, opponent_1, opponent_2, opponent_3, 4)
