from q_network import DQLAgent, ReplayBuffer, Q_network, State
from play import play_game, play_strategic_game
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

def train(training_episodes: int, num_players: int = 4):
    weights_dir = 'Weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    # weights_path = os.path.join(weights_dir, f'weights_{training_episodes}.pth')
    weights_path = os.path.join(weights_dir, f'weights_0.pth')
    torch.save(agent.q_network.state_dict(), weights_path)

    for _ in range(training_episodes):
        agent.q_network.load_state_dict(torch.load(weights_path))
        play_game(agent, num_players)
        weights_path = os.path.join(weights_dir, f'weights_{_ + 1}.pth')
        torch.save(agent.q_network.state_dict(), weights_path)

# def train(training_episodes: int, num_players: int = 4):
#     agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
#     torch.save(agent.q_network.state_dict(), f'weights_{training_episodes}.pth')
#     for _ in range(training_episodes):
#         agent.q_network.load_state_dict(torch.load(f'weights_{training_episodes}.pth'))
#         play_game(agent, num_players)
#         torch.save(agent.q_network.state_dict(), f'weights_{training_episodes}.pth')


def train_strategic(training_episodes: int, num_players: int = 4, opponent_1_train_iter: int = 90, 
                    opponent_2_train_iter: int = 85, opponent_3_train_iter: int = 80):
    
    '''
    Now, we want to train our model against earlier versions of itself. We need 3 opponents. 
    We want to train ours against 3 earlier versions of itself. 
    Train against the version that was trained for 90 episodes.
    Train against the version that was trained for 85 episodes.
    Train against the version that was trained for 80 episodes.
    '''
    
    weights_dir = 'Weights'
    weights_dir_strategic = 'Weights against itself'

    agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    agent_weights_path = os.path.join(weights_dir, f'weights_100.pth')
    
    # torch.save(agent.q_network.state_dict(), weights_path) -----> no need for this because these weights are ALREADY SAVED by training against random opponents 
    opponent_1 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    opponent_1_weights_path = os.path.join(weights_dir, f'weights_{opponent_1_train_iter}')
    opponent_1.q_network.load_state_dict(torch.load(opponent_1_weights_path))
    
    opponent_2 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    opponent_2_weights_path = os.path.join(weights_dir, f'weights_{opponent_2_train_iter}')
    opponent_2.q_network.load_state_dict(torch.load(opponent_2_weights_path))
    
    opponent_3 = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
    opponent_3_weights_path = os.path.join(weights_dir, f'weights_{opponent_3_train_iter}')
    opponent_3.q_network.load_state_dict(torch.load(opponent_3_weights_path))
    
    for _ in range(training_episodes):
        agent.q_network.load_state_dict(torch.load(agent_weights_path))
        play_strategic_game(agent, opponent_1, opponent_2, opponent_3, num_players)
        agent_weights_path = os.path.join(weights_dir_strategic, f'weights_{_ + 1}_strategic.pth')
        torch.save(agent.q_network.state_dict(), agent_weights_path)

