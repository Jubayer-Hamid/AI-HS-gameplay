import torch
import torch.nn as nn 
import pprint
import numpy as np
import random
from collections import deque, namedtuple
from typing import List

pp = pprint.PrettyPrinter()


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILON = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100


# dimensions of feature map for each training example in our model 
NUM_STATES = 546 # state vector size 
NUM_ACTIONS = 2048 # Number of possible combinations of bids 


class Q_network(nn.Module):
    '''
    Q_network will take in a state and return Q_values of all possible actions. 
    Input = array of size state_dim.
    Output = array of size action_dim. 
    '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Q_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class DQLAgent:
    '''
    DQL agent will explore using epsilon-greedy policy and then train the Q_network

    '''

    def __init__(self, state_size, action_size, hidden_size, batch_size, lr, gamma, eps_start, eps_end, eps_decay):

        # Store variables for the class
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.loss_fn = nn.MSELoss()
        self.tau = 0.001

        # Define the two Q-networks to be used: 
        self.q_network = Q_network(state_size, action_size, hidden_size)
        self.target_q_network = Q_network(state_size, action_size, hidden_size)
        
        # Initialise both of them to have the same weights 
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Disable dropout for target_q_network which is only used for evaluation 
        self.target_q_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # replay buffer: 
        self.memory = []
        
        # epsilon for epsilon-greedy algorithm to do exploration
        self.epsilon = self.eps_start

    def act(self, current_state, possible_moves):
        # use epsilon-greedy algorithm to do exploration
        
        # possible_moves is a np array of shape (1, action_size) 
        # with 0s for actions that are not possible and 1's for actions that are permissible 
        ######actual code:#####
        # if random.random() < self.epsilon:
        #     # make it randomly choose only those actions that are available 
        #     # Tip: keep action_size the same, just use some numpy function to only choose amongst certain parts of the vector 
        #     action = random.randrange(self.action_size) 
        
        # else:
        #     with torch.no_grad():  # We don't need gradients right now so do not need to build computation graphs 
        #         # q_network expects a batch of inputs. 
        #         state = torch.from_numpy(current_state).float().unsqueeze(0)
        #         q_values = self.q_network(state)
                
        #         mask = torch.tensor([0 if action in possible_moves else -float('inf') for action in range(NUM_ACTIONS)])
        #         q_values_masked = q_values + mask


        #         # choose action with the highest Q_value in this state 
        #         # action = q_values.argmax().item() 
                
        #         best_possible_action = np.argmax(q_values_masked).item()
        #         action = np.where(possible_moves)[0][best_possible_action]
        
        # return action  # an integer between 0 and action_size-1 
    ##########actual code#############
        if random.random() < self.epsilon:
            # make it randomly choose only those actions that are available
            # Tip: keep action_size the same, just use some numpy function to only choose amongst certain parts of the vector
            possible_actions = np.where(possible_moves)[0]
            action = np.random.choice(possible_actions)

        else:
            with torch.no_grad():  # We don't need gradients right now so do not need to build computation graphs
                # q_network expects a batch of inputs.
                state = torch.from_numpy(current_state).float().unsqueeze(0)
                q_values = self.q_network(state)

                # Mask the q_values based on possible_moves
                masked_q_values = np.where(possible_moves, q_values, np.nan)

                # choose action with the highest Q_value in this state
                best_possible_action = np.nanargmax(masked_q_values).item()
                action = np.where(possible_moves)[0][best_possible_action]

        return action  # an integer between 0 and action_size-1
    '''
    Suppose in total there are 5 possible actions: 
    actions = [a1, a2, a3, a4, a5]
    now suppose i choose action a2. I ONLY RETURN THE INDEX 1. This works becauese actions has fixed dimensions
    '''


    def train(self, batch):
        '''
        Train the Q_network using vanilla DQL algorithm
        '''
        states, actions, rewards, next_states, end_state = zip(*batch)

        # vertically stack the components so that each tensor has shape (batch_size, _) where _ could be state_dim, action_dim, or 1 (for rewards)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        end_state = torch.from_numpy(np.vstack(end_state).astype(np.uint8)).float()

        
        # Compute Q-values for current states and next states
        # Q_values of each state-action pair. 
        current_q_values = self.q_network(states).gather(1, actions) # tensor will have shape (batch_size, 1)

        # Q_value of next state and the best possible action from the next state. 
        next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1) # tensor will have shape (batch_size, 1)
                                                                           
        target_q_values = rewards + (self.gamma * next_q_values * (1 - end_state)) # tensor will have shape (batch_size, 1)

        # Compute the loss and update the q_network (NOT the target_q_network)
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # update target_q_network parameters 
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def update_epsilon(self, episode, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * (1 - episode / 200))
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "end_state"])

    def add(self, state, action, reward, next_state, end_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, end_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class State():
    def __init__(self, agent: DQLAgent) -> None:
        self.agent = agent

    def state_vector(self, revealed_card:str, deck: List[str], cards: List[str], money: List[int], player_2_cards: List[str],
                     player_2_money: List[int], player_3_cards: List[str], player_3_money: List[int],
                     player_4_money: List[int], player_4_cards: List[str]) -> np.ndarray:
        vec = []
        
        NUMBERED_CARDS = [f'{i}' for i in range(1,11)]
        SPECIAL_POSITIVE_CARDS = ['x2', 'x2', 'x2']
        SPECIAL_NEGATIVE_CARDS = ['1/2', '-5', 'discard']
        all_cards = []
        all_cards.extend(NUMBERED_CARDS)
        all_cards.extend(SPECIAL_POSITIVE_CARDS)
        all_cards.extend(SPECIAL_NEGATIVE_CARDS)
        
        money_notes = [1,2,3,4,6,8,10,12,15,20,25]
        
        # first element should be card shown on the table. 
        if revealed_card == '':
            vec.append(0)
        else:
            index = all_cards.index(revealed_card)
            vec.append(index + 1)

        # next 10 + 3 + 3 elements will be for cards remaining: 
        for i in range(1, 11):
            if f'{i}' in deck:
                vec.append(1)
            else:
                vec.append(0)
        
        for card in SPECIAL_POSITIVE_CARDS:
            if card in deck:
                vec.append(1)
            else:
                vec.append(0)
        
        for card in SPECIAL_NEGATIVE_CARDS:
            if card in deck:
                vec.append(1)
            else:
                vec.append(0)
        
        # 29 elements so far 
        
        cards_owned_by_player = cards
        for card in all_cards:
            if card in cards_owned_by_player:
                vec.append(1)
            else:
                vec.append(0)

        for note in money_notes:
            if note in money:
                vec.append(1)
            else:
                vec.append(0)


        cards_owned_by_player = player_2_cards
        for card in all_cards:
            if card in cards_owned_by_player:
                vec.append(1)
            else:
                vec.append(0)

        for note in money_notes:
            if note in player_2_money:
                vec.append(1)
            else:
                vec.append(0)

        cards_owned_by_player = player_3_cards
        for card in all_cards:
            if card in cards_owned_by_player:
                vec.append(1)
            else:
                vec.append(0)

        for note in money_notes:
            if note in player_3_money:
                vec.append(1)
            else:
                vec.append(0)

        cards_owned_by_player = player_4_cards
        for card in all_cards:
            if card in cards_owned_by_player:
                vec.append(1)
            else:
                vec.append(0)

        for note in money_notes:
            if note in player_4_money:
                vec.append(1)
            else:
                vec.append(0)

        
        return np.array(vec)
        # state dim = 29 + 3x16x11 = 557
    

    

