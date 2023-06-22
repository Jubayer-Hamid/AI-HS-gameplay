from typing import List
import random
from q_network import DQLAgent, ReplayBuffer, State




NUM_STATES = 546 # state vector size 
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


NUMBERED_CARDS = [f'{i}' for i in range(1,11)]
SPECIAL_POSITIVE_CARDS = ['x2', 'x2', 'x2']
SPECIAL_NEGATIVE_CARDS = ['1/2', '-5', 'discard']

def generate_bid_combinations(money):
        all_combinations = [[]]  # Start with an empty combination
        for num in money:
            combinations_with_num = []
            for comb in all_combinations:
                combinations_with_num.append(comb + [num])  # Add the current number to each existing combination
            # Extend the list with the new combinations - this means there are also combinations that do NOT include num 
            all_combinations.extend(combinations_with_num)  
        return all_combinations

def action_dictionary() -> dict:
    all_combinations = generate_bid_combinations([1,2,3,4,6,8,10,12,15,20,25])
    dictionary = {}
    for i in range(len(all_combinations)):
        dictionary[i] = all_combinations[i]
    return dictionary

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Value not found



class Player():
    def __init__(self, name: str) -> None:
        '''
        Initialize player. Player initially has only money
        '''
        self.name = name 
        
        self.money = [1,2,3,4,6,8,10,12,15,20,25]
        self.special_cards = []
        self.number_cards = []
        
    def calculate_score(self) -> float:
        score = 0
        for elem in self.number_cards:
            score += int(elem)
        if '-5' in self.special_cards:
            score -= 5
        if 'x2' in self.special_cards:
            score *= 2
        if '1/2' in self.special_cards:
            score *= 0.5
        return score     
    
    def generate_bid_combinations(self, money):
        all_combinations = [[]]  # Start with an empty combination
        for num in money:
            combinations_with_num = []
            for comb in all_combinations:
                combinations_with_num.append(comb + [num])  # Add the current number to each existing combination
            # Extend the list with the new combinations - this means there are also combinations that do NOT include num 
            all_combinations.extend(combinations_with_num)  
        return all_combinations


    def make_random_bid_for_positive_card(self, current_bid:int, money_left:List[int]) -> List[int]:
        bids = self.generate_bid_combinations(money_left) # list of combinations of money cards (only money cards the player still has)
        possible_bids = [bid for bid in bids if sum(bid) > current_bid] # only those bids are feasible that beat the current bid
        # you can also pass:
        possible_bids.append([])
        return random.choice(possible_bids)

    
    '''
    def make_intelligent_bid(self) -> int:
        
        # Make bid using AI model     
    '''

class Game():
    def __init__(self, agent: DQLAgent, num_players: int) -> None:
        
        '''
        Our agent is the only intelligent player in this version of the game. Opponents are going to make moves randomly.
        '''
        
        # Convention : Player 0 will be our AI player 
        self.players = []
        for i in range(num_players):
            player = Player(f'Player {i}')
            self.players.append(player)
        
        # Initialise deck of cards 
        self.deck = []
        self.deck.extend(NUMBERED_CARDS)
        self.deck.extend(SPECIAL_POSITIVE_CARDS)
        self.deck.extend(SPECIAL_NEGATIVE_CARDS)
        
        # Shuffle deck 
        random.shuffle(self.deck)

        # Current revealed card on deck: 
        self.revealed_card = ''

        # self.agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
        self.agent = agent

        # Create an instance of the ReplayBuffer class
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

        # action dictionary for help:
        self.action_dictionary = action_dictionary()
        # all possible bids:
        self.all_possible_bids = generate_bid_combinations([1,2,3,4,6,8,10,12,15,20,25])
    # Coding sequentially as a game would progress

    # First, we reveal a card from deck: 
    def reveal_deck_card(self) -> str:
        '''
        returns a string with the card that's revealed.
        Even if it's a number card, the revealed card would be in a string
        '''
        self.revealed_card = random.choice(self.deck)
        return self.revealed_card
    
    # Next, players bid for the revealed card 
    def bidding_round_train(self, revealed_card:str, starting_player: Player) -> None:
        # rotate list of players:
        index = self.players.index(starting_player)
        self.players_copy = self.players[index:] + self.players[:index]

        # First, we init a list of possible bidders - if a player has no money, they can't bid
        # As the bidding goes, we will get rid of bidders 
        bidders = list(self.players_copy)    
        bids_of_each_player = {}
        for player in bidders:
            bids_of_each_player[player] = []        
        
        # episode count:
        episode = 0
        # Make everyone bid again and again until one bidder is left         
        current_bid = 0 # the bid to beat 
        while len(bidders) > 1:
            for player in bidders:
                # consider the actions of each bidder 
                # bidder makes decision. If AI player, then intelligent. Otherwise, random 
                episode += 1
            
                if player.name == 'Player 0':
                    state_prime = State(self.agent)
                    state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                               self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                               self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                               self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                               self.players[3].money)
                    
                    # def state_vector(self, revealed_card:str, deck: List[str], cards: List[str], money: List[int], player_2_cards: List[str],
                    #  player_2_money: List[int], player_3_cards: List[str], player_3_money: List[int],
                    #  player_4_money: List[int], player_4_cards: List[str]) -> np.ndarray:
                    # money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    
                    possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                    for i, bid in enumerate(self.all_possible_bids):
                        if all(elem in player.money for elem in bid):
                            # key = find_key(self.action_dictionary, bid)
                            possible_moves[i] = 1

                    # for bid in self.all_possible_bids:
                    #     if all(elem in player.money for elem in bid):
                    #         # if not check_common_elements(bid, bids_of_each_player[player]):
                    #             key = find_key(self.action_dictionary, bid)
                    #             possible_moves.append(key)
                    
                    bid_number = self.agent.act(state, possible_moves)
                    action = bid_number
                    bid = self.action_dictionary[bid_number]
                
                else:
                    # player had already bidded some money before. they need to meet the current bid - what they already paid 
                    money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    bid = player.make_random_bid_for_positive_card(current_bid - sum(bids_of_each_player[player]), money_left)
                
                # the sum of the money they have bidded at THIS time - not the total bid they made for the card. If they bid 0 
                # this time, then that means they pulled out 
                total_bid = sum(bid)
                
                # If they bidded no money, they are pulling out 
                if total_bid == 0:
                    if revealed_card in SPECIAL_NEGATIVE_CARDS:
                        player.money.extend(bids_of_each_player[player]) # this player gets the money back + the negative card 
                        player.special_cards.append(revealed_card)
                        # need to end bidding right now 
                        for player_prime in bidders:
                            if player_prime != player:
                                bidders.remove(player_prime)
                    else:
                        player.money.extend(bids_of_each_player[player])
                        bidders.remove(player)
                            
                # they bidded SOME money - they are IN 
                else:
                    bids_of_each_player[player].extend(bid)
                    for _ in bid:
                        player.money.remove(_) # they lose the money they bidded. If they lose the bidding war, they are gonna get the money back 
                    
                    current_bid = total_bid
            
            if player.name == 'Player 0':
                state_prime = State(self.agent)
                next_state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                                self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                                self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                                self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                                self.players[3].money)
                reward = 0 
                if len(self.deck) <= 1:
                    end_state = 1
                else:
                    end_state = 0 
                self.memory.add(state, action, reward, next_state, end_state)
                if len(self.memory) < batch_size:
                    continue
                else:
                    batch = self.memory.sample(batch_size)

                    # Train the Q-network using the sampled batch of experience tuples
                    self.agent.train(batch)

                    # Update the exploration parameter epsilon
                    self.agent.update_epsilon(episode, min_epsilon)
        
        
        # someone won the bid by now
        # if it's a positive card:
        winning_player = bidders[0]
        if revealed_card in SPECIAL_NEGATIVE_CARDS:
            pass
        else:
            if revealed_card in SPECIAL_POSITIVE_CARDS:
                winning_player.special_cards.append(revealed_card)
            else:
                winning_player.number_cards.append(revealed_card)

            # rest of the players must get back their money
            for player_prime in self.players:
                if player_prime != winning_player:
                    player_prime.money.extend(bids_of_each_player[player_prime])
        
        
        self.deck.remove(revealed_card)
    
    def bidding_round(self, revealed_card:str, starting_player: Player) -> None:
        # rotate list of players:
        index = self.players.index(starting_player)
        self.players_copy = self.players[index:] + self.players[:index]

        # First, we init a list of possible bidders - if a player has no money, they can't bid
        # As the bidding goes, we will get rid of bidders 
        bidders = list(self.players_copy)    
        bids_of_each_player = {}
        for player in bidders:
            bids_of_each_player[player] = []        
        
        # episode count:
        episode = 0
        # Make everyone bid again and again until one bidder is left         
        current_bid = 0 # the bid to beat 
        while len(bidders) > 1:
            for player in bidders:
                # consider the actions of each bidder 
                # bidder makes decision. If AI player, then intelligent. Otherwise, random 
                episode += 1
            
                if player.name == 'Player 0':
                    state_prime = State(self.agent)
                    state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                               self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                               self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                               self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                               self.players[3].money)
                    
                    # def state_vector(self, revealed_card:str, deck: List[str], cards: List[str], money: List[int], player_2_cards: List[str],
                    #  player_2_money: List[int], player_3_cards: List[str], player_3_money: List[int],
                    #  player_4_money: List[int], player_4_cards: List[str]) -> np.ndarray:
                    # money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    
                    possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                    for i, bid in enumerate(self.all_possible_bids):
                        if all(elem in player.money for elem in bid):
                            # key = find_key(self.action_dictionary, bid)
                            possible_moves[i] = 1

                    # for bid in self.all_possible_bids:
                    #     if all(elem in player.money for elem in bid):
                    #         # if not check_common_elements(bid, bids_of_each_player[player]):
                    #             key = find_key(self.action_dictionary, bid)
                    #             possible_moves.append(key)
                    
                    bid_number = self.agent.act(state, possible_moves)
                    action = bid_number
                    bid = self.action_dictionary[bid_number]
                
                else:
                    # player had already bidded some money before. they need to meet the current bid - what they already paid 
                    money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    bid = player.make_random_bid_for_positive_card(current_bid - sum(bids_of_each_player[player]), money_left)
                
                # the sum of the money they have bidded at THIS time - not the total bid they made for the card. If they bid 0 
                # this time, then that means they pulled out 
                total_bid = sum(bid)
                
                # If they bidded no money, they are pulling out 
                if total_bid == 0:
                    if revealed_card in SPECIAL_NEGATIVE_CARDS:
                        player.money.extend(bids_of_each_player[player]) # this player gets the money back + the negative card 
                        player.special_cards.append(revealed_card)
                        # need to end bidding right now 
                        for player_prime in bidders:
                            if player_prime != player:
                                bidders.remove(player_prime)
                    else:
                        player.money.extend(bids_of_each_player[player])
                        bidders.remove(player)
                            
                # they bidded SOME money - they are IN 
                else:
                    bids_of_each_player[player].extend(bid)
                    for _ in bid:
                        player.money.remove(_) # they lose the money they bidded. If they lose the bidding war, they are gonna get the money back 
                    
                    current_bid = total_bid
            
            # if player.name == 'Player 0':
            #     state_prime = State(self.agent)
            #     next_state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
            #                                     self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
            #                                     self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
            #                                     self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
            #                                     self.players[3].money)
            #     reward = 0 
            #     if len(self.deck) <= 1:
            #         end_state = 1
            #     else:
            #         end_state = 0 
            #     self.memory.add(state, action, reward, next_state, end_state)
            #     if len(self.memory) < batch_size:
            #         continue
            #     else:
            #         batch = self.memory.sample(batch_size)

            #         # Train the Q-network using the sampled batch of experience tuples
            #         self.agent.train(batch)

            #         # Update the exploration parameter epsilon
            #         self.agent.update_epsilon(episode, min_epsilon)
        
        # someone won the bid by now
        # if it's a positive card:
        winning_player = bidders[0]
        if revealed_card in SPECIAL_NEGATIVE_CARDS:
            pass
        else:
            if revealed_card in SPECIAL_POSITIVE_CARDS:
                winning_player.special_cards.append(revealed_card)
            else:
                winning_player.number_cards.append(revealed_card)

            # rest of the players must get back their money
            for player_prime in self.players:
                if player_prime != winning_player:
                    player_prime.money.extend(bids_of_each_player[player_prime])
        
        
        self.deck.remove(revealed_card)

    

class StrategicGame():
    def __init__(self, agent: DQLAgent, opponent_1: DQLAgent, opponent_2: DQLAgent, opponent_3: DQLAgent, num_players: int=4) -> None:
        
        '''
        Our agent is the only intelligent player in this version of the game. Opponents are going to make moves randomly.
        '''
        
        # Convention : Player 0 will be our AI player 
        self.players = []
        for i in range(num_players):
            player = Player(f'Player {i}')
            self.players.append(player)
        
        # Initialise deck of cards 
        self.deck = []
        self.deck.extend(NUMBERED_CARDS)
        self.deck.extend(SPECIAL_POSITIVE_CARDS)
        self.deck.extend(SPECIAL_NEGATIVE_CARDS)
        
        # Shuffle deck 
        random.shuffle(self.deck)

        # Current revealed card on deck: 
        self.revealed_card = ''

        # self.agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)
        self.agent = agent
        self.opponent_1 = opponent_1
        self.opponent_2 = opponent_2
        self.opponent_3 = opponent_3
        
        # Create an instance of the ReplayBuffer class
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

        # action dictionary for help:
        self.action_dictionary = action_dictionary()
        # all possible bids:
        self.all_possible_bids = generate_bid_combinations([1,2,3,4,6,8,10,12,15,20,25])
    # Coding sequentially as a game would progress

    # First, we reveal a card from deck: 
    def reveal_deck_card(self) -> str:
        '''
        returns a string with the card that's revealed.
        Even if it's a number card, the revealed card would be in a string
        '''
        self.revealed_card = random.choice(self.deck)
        return self.revealed_card
    
    # Next, players bid for the revealed card 
    def bidding_round_train(self, revealed_card:str, starting_player: Player) -> None:
        # rotate list of players:
        index = self.players.index(starting_player)
        self.players_copy = self.players[index:] + self.players[:index]

        # First, we init a list of possible bidders - if a player has no money, they can't bid
        # As the bidding goes, we will get rid of bidders 
        bidders = list(self.players_copy)    
        bids_of_each_player = {}
        for player in bidders:
            bids_of_each_player[player] = []        
        
        # episode count:
        episode = 0
        # Make everyone bid again and again until one bidder is left         
        current_bid = 0 # the bid to beat 
        while len(bidders) > 1:
            for player in bidders:
                # consider the actions of each bidder 
                # bidder makes decision. If AI player, then intelligent. Otherwise, random 
                episode += 1
            
                if player.name == 'Player 0':
                    state_prime = State(self.agent)
                    state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                               self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                               self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                               self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                               self.players[3].money)
                    
                    # def state_vector(self, revealed_card:str, deck: List[str], cards: List[str], money: List[int], player_2_cards: List[str],
                    #  player_2_money: List[int], player_3_cards: List[str], player_3_money: List[int],
                    #  player_4_money: List[int], player_4_cards: List[str]) -> np.ndarray:
                    # money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    
                    possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                    for i, bid in enumerate(self.all_possible_bids):
                        if all(elem in player.money for elem in bid):
                            # key = find_key(self.action_dictionary, bid)
                            possible_moves[i] = 1

                    # for bid in self.all_possible_bids:
                    #     if all(elem in player.money for elem in bid):
                    #         # if not check_common_elements(bid, bids_of_each_player[player]):
                    #             key = find_key(self.action_dictionary, bid)
                    #             possible_moves.append(key)
                    
                    bid_number = self.agent.act(state, possible_moves)
                    action = bid_number
                    bid = self.action_dictionary[bid_number]
                

                else:
                    # Find player number: 
                    i = int(player.name[-1])
                    
                    
                    # now we have player number. access agent number. 
                    if i == 1:
                        opponent = self.opponent_1
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[2].number_cards + self.players[2].special_cards, self.players[2].money,
                                                         self.players[3].number_cards + self.players[3].special_cards, self.players[3].money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_1.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]
                    
                    elif i == 2:
                        opponent = self.opponent_2
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[3].number_cards + self.players[3].special_cards, self.players[3].money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money,
                                                         self.players[1].number_cards + self.players[1].special_cards, self.players[1].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_2.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]
                    
                    elif i == 3:
                        opponent = self.opponent_3
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money,
                                                         self.players[1].number_cards + self.players[1].special_cards, self.players[1].money,
                                                         self.players[2].number_cards + self.players[2].special_cards, self.players[2].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_3.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]

                # the sum of the money they have bidded at THIS time - not the total bid they made for the card. If they bid 0 
                # this time, then that means they pulled out 
                total_bid = sum(bid)
                
                # If they bidded no money, they are pulling out 
                if total_bid == 0:
                    if revealed_card in SPECIAL_NEGATIVE_CARDS:
                        player.money.extend(bids_of_each_player[player]) # this player gets the money back + the negative card 
                        player.special_cards.append(revealed_card)
                        # need to end bidding right now 
                        for player_prime in bidders:
                            if player_prime != player:
                                bidders.remove(player_prime)
                    else:
                        player.money.extend(bids_of_each_player[player])
                        bidders.remove(player)
                            
                # they bidded SOME money - they are IN 
                else:
                    bids_of_each_player[player].extend(bid)
                    for _ in bid:
                        player.money.remove(_) # they lose the money they bidded. If they lose the bidding war, they are gonna get the money back 
                    
                    current_bid = total_bid
            
            # We only train player 0 - our agent. 
            if player.name == 'Player 0':
                state_prime = State(self.agent)
                next_state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                                self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                                self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                                self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                                self.players[3].money)
                reward = 0 
                if len(self.deck) <= 1:
                    end_state = 1
                else:
                    end_state = 0 
                self.memory.add(state, action, reward, next_state, end_state)
                if len(self.memory) < batch_size:
                    continue
                else:
                    batch = self.memory.sample(batch_size)

                    # Train the Q-network using the sampled batch of experience tuples
                    self.agent.train(batch)

                    # Update the exploration parameter epsilon
                    self.agent.update_epsilon(episode, min_epsilon)
        
        
        # someone won the bid by now
        # if it's a positive card:
        winning_player = bidders[0]
        if revealed_card in SPECIAL_NEGATIVE_CARDS:
            pass
        else:
            if revealed_card in SPECIAL_POSITIVE_CARDS:
                winning_player.special_cards.append(revealed_card)
            else:
                winning_player.number_cards.append(revealed_card)

            # rest of the players must get back their money
            for player_prime in self.players:
                if player_prime != winning_player:
                    player_prime.money.extend(bids_of_each_player[player_prime])
        
        
        self.deck.remove(revealed_card)
    

    def bidding_round(self, revealed_card:str, starting_player: Player) -> None:
        # rotate list of players:
        index = self.players.index(starting_player)
        self.players_copy = self.players[index:] + self.players[:index]

        # First, we init a list of possible bidders - if a player has no money, they can't bid
        # As the bidding goes, we will get rid of bidders 
        bidders = list(self.players_copy)    
        bids_of_each_player = {}
        for player in bidders:
            bids_of_each_player[player] = []        
        
        # episode count:
        episode = 0
        # Make everyone bid again and again until one bidder is left         
        current_bid = 0 # the bid to beat 
        while len(bidders) > 1:
            for player in bidders:
                # consider the actions of each bidder 
                # bidder makes decision. If AI player, then intelligent. Otherwise, random 
                episode += 1
            
                if player.name == 'Player 0':
                    state_prime = State(self.agent)
                    state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
                                               self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
                                               self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
                                               self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
                                               self.players[3].money)
                    
                    # def state_vector(self, revealed_card:str, deck: List[str], cards: List[str], money: List[int], player_2_cards: List[str],
                    #  player_2_money: List[int], player_3_cards: List[str], player_3_money: List[int],
                    #  player_4_money: List[int], player_4_cards: List[str]) -> np.ndarray:
                    # money_left = [x for x in player.money if x not in bids_of_each_player[player]]
                    
                    possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                    for i, bid in enumerate(self.all_possible_bids):
                        if all(elem in player.money for elem in bid):
                            # key = find_key(self.action_dictionary, bid)
                            possible_moves[i] = 1

                    # for bid in self.all_possible_bids:
                    #     if all(elem in player.money for elem in bid):
                    #         # if not check_common_elements(bid, bids_of_each_player[player]):
                    #             key = find_key(self.action_dictionary, bid)
                    #             possible_moves.append(key)
                    
                    bid_number = self.agent.act(state, possible_moves)
                    action = bid_number
                    bid = self.action_dictionary[bid_number]
                

                else:
                    # Find player number: 
                    i = int(player.name[-1])
                    
                    
                    # now we have player number. access agent number. 
                    if i == 1:
                        opponent = self.opponent_1
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[2].number_cards + self.players[2].special_cards, self.players[2].money,
                                                         self.players[3].number_cards + self.players[3].special_cards, self.players[3].money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_1.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]
                    
                    elif i == 2:
                        opponent = self.opponent_2
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[3].number_cards + self.players[3].special_cards, self.players[3].money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money,
                                                         self.players[1].number_cards + self.players[1].special_cards, self.players[1].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_2.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]
                    
                    elif i == 3:
                        opponent = self.opponent_3
                        state_prime = State(opponent)
                        state = state_prime.state_vector(revealed_card, self.deck, player.number_cards + player.special_cards, player.money,
                                                         self.players[0].number_cards + self.players[0].special_cards, self.players[0].money,
                                                         self.players[1].number_cards + self.players[1].special_cards, self.players[1].money,
                                                         self.players[2].number_cards + self.players[2].special_cards, self.players[2].money)
                        
                        possible_moves = [0 for _ in range(len(self.all_possible_bids))]
                        for i, bid in enumerate(self.all_possible_bids):
                            if all(elem in player.money for elem in bid):
                                # key = find_key(self.action_dictionary, bid)
                                possible_moves[i] = 1

                        # for bid in self.all_possible_bids:
                        #     if all(elem in player.money for elem in bid):
                        #         # if not check_common_elements(bid, bids_of_each_player[player]):
                        #             key = find_key(self.action_dictionary, bid)
                        #             possible_moves.append(key)
                        
                        bid_number = self.opponent_3.act(state, possible_moves)
                        action = bid_number
                        bid = self.action_dictionary[bid_number]

                # the sum of the money they have bidded at THIS time - not the total bid they made for the card. If they bid 0 
                # this time, then that means they pulled out 
                total_bid = sum(bid)
                
                # If they bidded no money, they are pulling out 
                if total_bid == 0:
                    if revealed_card in SPECIAL_NEGATIVE_CARDS:
                        player.money.extend(bids_of_each_player[player]) # this player gets the money back + the negative card 
                        player.special_cards.append(revealed_card)
                        # need to end bidding right now 
                        for player_prime in bidders:
                            if player_prime != player:
                                bidders.remove(player_prime)
                    else:
                        player.money.extend(bids_of_each_player[player])
                        bidders.remove(player)
                            
                # they bidded SOME money - they are IN 
                else:
                    bids_of_each_player[player].extend(bid)
                    for _ in bid:
                        player.money.remove(_) # they lose the money they bidded. If they lose the bidding war, they are gonna get the money back 
                    
                    current_bid = total_bid
            
            # # We only train player 0 - our agent. 
            # if player.name == 'Player 0':
            #     state_prime = State(self.agent)
            #     next_state = state_prime.state_vector(revealed_card, self.deck, self.players[0].number_cards + self.players[0].special_cards, 
            #                                     self.players[0].money, self.players[1].number_cards + self.players[1].special_cards, 
            #                                     self.players[1].money, self.players[2].number_cards + self.players[2].special_cards, 
            #                                     self.players[2].money, self.players[3].number_cards + self.players[3].special_cards, 
            #                                     self.players[3].money)
            #     reward = 0 
            #     if len(self.deck) <= 1:
            #         end_state = 1
            #     else:
            #         end_state = 0 
            #     self.memory.add(state, action, reward, next_state, end_state)
            #     if len(self.memory) < batch_size:
            #         continue
            #     else:
            #         batch = self.memory.sample(batch_size)

            #         # Train the Q-network using the sampled batch of experience tuples
            #         self.agent.train(batch)

            #         # Update the exploration parameter epsilon
            #         self.agent.update_epsilon(episode, min_epsilon)
    
        # someone won the bid by now
        # if it's a positive card:
        winning_player = bidders[0]
        if revealed_card in SPECIAL_NEGATIVE_CARDS:
            pass
        else:
            if revealed_card in SPECIAL_POSITIVE_CARDS:
                winning_player.special_cards.append(revealed_card)
            else:
                winning_player.number_cards.append(revealed_card)

            # rest of the players must get back their money
            for player_prime in self.players:
                if player_prime != winning_player:
                    player_prime.money.extend(bids_of_each_player[player_prime])
        
        
        self.deck.remove(revealed_card)
    