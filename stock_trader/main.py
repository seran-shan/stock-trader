'''
This is the main file for the stock trading trainer.
'''
from typing import Any
import argparse
import yaml
import yfinance as yf
import pandas as pd

from environments.stock_trading_env import StockTradingEnv


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
    parser.add_argument('--render_mode', type=str, default='human', help='Render mode for the environment (e.g., human, none)')
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

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False

        while not done:
            # Random action as an example, replace with DDQN decided action
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)

            if truncated:
                break

        print(f"Episode: {episode + 1}, Total Reward: {info['total_reward']}, Total Profit: {info['total_profit']}")

    env.close()

def main() -> None:
    '''
    The main function for the stock trading trainer.
    '''
    args = parse_arguments()
    stock_ticker = args[0]
    render_mode = args[1]
    config = load_config()
    run_environment(config, stock_ticker, render_mode)

if __name__ == "__main__":
    main()
