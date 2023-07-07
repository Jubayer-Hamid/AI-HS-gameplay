# Code for the main game 
import random
from game import Game, Player, StrategicGame
from q_network import DQLAgent

def play_game(agent, num_players):
    game = Game(agent, num_players)

    starting_player_index = random.randint(0, num_players-1)
    
    # start game
    while len(game.deck) > 0:
        
        revealed_card = game.reveal_deck_card() # this is a string 
        game.bidding_round_train(revealed_card, game.players[starting_player_index])
        starting_player_index += 1
        starting_player_index = starting_player_index % num_players
    # game over

    # calculate score: 
    player_score_dict = {}
    for player in game.players:
        score = player.calculate_score()
        player_score_dict[player] = score
    winning_player = max(player_score_dict, key=player_score_dict.get)
    if winning_player == 'Player 0':
        print('SUCCESS: AI Model has won!')
    else:
        print('FAILURE: AI Model has lost.')

def play_game_eval(agent, num_players):
    game = Game(agent, num_players)

    starting_player_index = random.randint(0, num_players-1)
    
    # start game
    while len(game.deck) > 0:
        
        revealed_card = game.reveal_deck_card() # this is a string 
        game.bidding_round(revealed_card, game.players[starting_player_index])
        starting_player_index += 1
        starting_player_index = starting_player_index % num_players
    # game over

    # calculate score: 
    player_score_dict = {}
    for player in game.players:
        score = player.calculate_score()
        player_score_dict[player] = score
    winning_player = max(player_score_dict, key=player_score_dict.get)
    if winning_player == 'Player 0':
        print('SUCCESS: AI Model has won!')
    else:
        print('FAILURE: AI Model has lost.')


def play_strategic_game(agent, opponent_1, opponent_2, opponent_3, num_players):
    game = StrategicGame(agent, opponent_1, opponent_2, opponent_3, num_players)

    starting_player_index = random.randint(0, num_players-1)
    
    # start game
    while len(game.deck) > 0:
        revealed_card = game.reveal_deck_card() # this is a string 
        game.bidding_round_train(revealed_card, game.players[starting_player_index])
        starting_player_index += 1
        starting_player_index = starting_player_index % num_players
    # game over

    # calculate score: 
    player_score_dict = {}
    for player in game.players:
        score = player.calculate_score()
        player_score_dict[player] = score
    
    winning_player = max(player_score_dict, key=player_score_dict.get)
    if winning_player == 'Player 0':
        print('SUCCESS: AI Model has won against earlier versions of itself!')
    else:
        print('FAILURE: AI Model has lost against earlier versions of itself.')


def play_strategic_game_eval(agent, opponent_1, opponent_2, opponent_3, num_players):
    game = StrategicGame(agent, opponent_1, opponent_2, opponent_3, num_players)

    starting_player_index = random.randint(0, num_players-1)
    
    # start game
    while len(game.deck) > 0:
        revealed_card = game.reveal_deck_card() # this is a string 
        game.bidding_round(revealed_card, game.players[starting_player_index])
        starting_player_index += 1
        starting_player_index = starting_player_index % num_players
    # game over

    # calculate score: 
    player_score_dict = {}
    for player in game.players:
        score = player.calculate_score()
        player_score_dict[player] = score
    
    winning_player = max(player_score_dict, key=player_score_dict.get)
    if winning_player == 'Player 0':
        print('SUCCESS: AI Model has won against earlier versions of itself!')
    else:
        print('FAILURE: AI Model has lost against earlier versions of itself.')

