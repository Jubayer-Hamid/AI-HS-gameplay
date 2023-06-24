import argparse
from train import train, train_strategic
from eval import eval_against_random_opponents, eval_against_strategic_opponents

# def main():
#     num_players = 4 
#     train(100, num_players)
#     train_strategic(100, 4, 90, 85, 80)
#     eval_against_random_opponents()
# if __name__ == '__main__':
#     main()

def main(train_mode, strategic_mode, num_players, episodes):
    if train_mode:
        train(episodes, num_players)
        train_strategic(episodes, num_players, episodes-5, episodes-10, episodes-15)
            
    else:
        if strategic_mode:
            eval_against_strategic_opponents(episodes, num_players)
        else:
            eval_against_random_opponents(episodes, num_players)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate agent for board game')
    parser.add_argument('mode', choices=['train', 'eval'], help='Mode: train or eval')
    parser.add_argument('--strategic', action='store_true', help='Enable strategic mode')
    parser.add_argument('--num_players', type=int, default=4, help='Number of players')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    args = parser.parse_args()

    if args.mode == 'train':
        main(True, args.strategic, args.num_players, args.episodes)
    elif args.mode == 'eval':
        main(False, args.strategic, args.num_players, args.episodes)

    # python script.py train --strategic --num_players 4 --episodes 100

    # python script.py eval --strategic --num_players 4 --episodes 100 
    # python script.py eval --num_players 4 --episodes 100



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train or evaluate agent for board game')
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument('--train', action='store_true', help='Train the agent')
#     group.add_argument('--eval', action='store_true', help='Evaluate the agent')
#     args = parser.parse_args()
    
#     if args.train:
#         main(True)
#     elif args.eval:
#         main(False)
#     else:
#         parser.print_help()
