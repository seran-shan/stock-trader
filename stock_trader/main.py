'''
This is the main file for the stock trading trainer.
'''
from typing import Any
import argparse
from termcolor import colored
import yaml
import yfinance as yf
import pandas as pd

from environments.stock_trading_env import StockTradingEnv
from stock_trader.agents.ddqn import DDQN


def parse_arguments() -> tuple[str, str]:
    """
    Parse command line arguments.

    Returns
    -------
    tuple[str, str]
        The stock ticker to train on and the render mode for the environment.
    """
    parser = argparse.ArgumentParser(description="Stock Trading Trainer")
    parser.add_argument('--stock_ticker', type=str, default="AAPL", help='Stock ticker to train on (e.g., AAPL, MSFT)')
    parser.add_argument('--render_mode', type=str, default=None, help='Render mode for the environment (e.g., human, none)')
    args = parser.parse_args()
    return args.stock_ticker, args.render_mode

def load_config() -> dict[str, Any]:
    '''
    Load the configuration for the stock trading environment.

    Returns
    -------
    config : dict[str, Any]
        The configuration for the stock trading environment.
    '''
    with open('stock_trader/config.yaml', 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_data(stock_ticker: str) -> pd.DataFrame:
    '''
    Load the data for the stock trading environment.

    Parameters
    ----------
    stock_ticker : str
        The stock ticker to train on.

    Returns
    -------
    data : pd.DataFrame
        The data for the stock trading environment.
    '''
    data = yf.download(stock_ticker, start='2020-01-01', end='2021-01-01')
    return data

def run_environment(config: dict[str, Any], stock_ticker: str, render_mode: str) -> None:
    """
    Run the stock trading environment.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration for the stock trading environment.
    stock_ticker : str
        The stock ticker to train on.
    render_mode : str
        The render mode for the environment.
    """
    df = load_data(stock_ticker)
    window_size = config['window_size']
    frame_bound = (window_size, len(df))
    num_episodes = config['num_episodes']

    env = StockTradingEnv(df, config['window_size'], frame_bound, render_mode)

    # Initialize DDQN agent
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DDQN(state_size, action_size, config["agent"])

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False

        while not done:
            # Flatten the observation to fit the DDQN input
            observation_flat = observation.flatten()

            # Random action as an example, replace with DDQN decided action
            action = agent.act(observation_flat)
            next_observation, reward, done, truncated, info = env.step(action)

            # Use a high placeholder error for new experiences
            placeholder_error = 1.0
            # Store the experience in the replay buffer
            agent.memory.add(error=placeholder_error, sample=(observation_flat, action, reward, next_observation.flatten(), done))

            agent.replay()

            # Update the target network
            # if episeode % config["agent"]["update_target_network_frequency"] == 0:
            agent.update_target_network()

            observation = next_observation

            if truncated:
                break

        print(colored(f"===== Episode Summary =====", 'cyan'))
        print(colored(f"Episode: {episode + 1}", 'cyan'))
        print(colored(f"Total Reward: {info['total_reward']:.2f}", 'magenta'))
        print(colored(f"Total Profit: {info['total_profit']:.2f}", 'blue'))
        print(colored("==========================================", 'cyan'))

    env.close()

def download_data(stock_ticker: str) -> None:
    '''
    Download the data for the stock trading environment.

    Parameters
    ----------
    stock_ticker : str
        The stock ticker to train on.
    '''
    data = yf.download(stock_ticker, start='2020-01-01', end='2021-01-01')
    data.to_csv(f'stock_trader/data/{stock_ticker}.csv')

def main() -> None:
    '''
    The main function for the stock trading trainer.
    '''
    stock_ticker, render_mode = parse_arguments()
    config = load_config()
    run_environment(config, stock_ticker, render_mode)

    # Uncomment to download data
    # download_data('AAPL')

if __name__ == "__main__":
    main()
