"""
This is the main file for the stock trading trainer.
"""
from typing import Any
import argparse
import signal
from termcolor import colored
import yaml
import yfinance as yf
import pandas as pd

# pylint: disable=import-error
from environments.stock_trading_env import StockTradingEnv
from stock_trader.agents.ddqn import DDQN


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The stock ticker to train on, the render mode for the environment, and the training flag.
    """
    parser = argparse.ArgumentParser(description="Stock Trading Trainer")
    parser.add_argument(
        "--stock_ticker",
        type=str,
        default="AAPL",
        help="Stock ticker to train on (e.g., AAPL, MSFT)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Render mode for the environment (e.g., human, none)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model if this flag is set, otherwise load the model",
    )
    args = parser.parse_args()
    return args


def load_config() -> dict[str, Any]:
    """
    Load the configuration for the stock trading environment.

    Returns
    -------
    config : dict[str, Any]
        The configuration for the stock trading environment.
    """
    with open("stock_trader/config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(stock_ticker: str) -> pd.DataFrame:
    """
    Load the data for the stock trading environment.

    Parameters
    ----------
    stock_ticker : str
        The stock ticker to train on.

    Returns
    -------
    data : pd.DataFrame
        The data for the stock trading environment.
    """
    data = yf.download(stock_ticker, start="2020-01-01", end="2021-01-01")
    return data


def train(config, stock_ticker, render_mode):
    """
    Train the model.

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
    window_size = config["window_size"]
    frame_bound = (window_size, len(df))
    num_episodes = config["num_episodes"]

    env = StockTradingEnv(df, config["window_size"], frame_bound, render_mode)

    # Initialize DDQN agent
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DDQN(state_size, action_size, config["agent"])

    def save_model(sig, frame):
        print("Saving model due to interruption...")
        agent.save_model(f'{config["agent"]["model_path"]}interrupted_model.h5')
        exit(0)

    signal.signal(signal.SIGINT, save_model)

    for episode in range(num_episodes):
        total_reward = 0
        total_profit = 0
        observation, info = env.reset()
        done = False

        while not done:
            # Flatten the observation to fit the DDQN input
            observation_flat = observation.flatten()

            # Random action as an example, replace with DDQN decided action
            action = agent.act(observation_flat)
            next_observation, reward, done, truncated, info = env.step(action)

            # Calculate the TD error
            td_error = agent.compute_td_error_from_experience(
                observation_flat, action, reward, next_observation.flatten(), done
            )
            # Store the experience in the replay buffer
            agent.memory.add(
                error=td_error,
                sample=(
                    observation_flat,
                    action,
                    reward,
                    next_observation.flatten(),
                    done,
                ),
            )

            agent.replay()

            # Update the target network
            # if episeode % config["agent"]["update_target_network_frequency"] == 0:
            agent.update_target_network()
            observation = next_observation

            total_reward += reward
            total_profit = info.get("total_profit", total_profit)

            if truncated:
                break

        if (episode + 1) % config["save_interval"] == 0:
            agent.save_model(f'{config["agent"]["model_path"]}model_{episode + 1}.h5')

        print(colored(f"===== Episode {episode + 1} Summary =====", "cyan"))
        print(colored(f"Total Reward: {total_reward:.2f}", "magenta"))
        print(colored(f"Total Profit: {total_profit:.2f}", "blue"))
        print(colored("==========================================", "cyan"))

    env.close()


def evaluate(config, stock_ticker, render_mode):
    """
    Evaluate the trained model.

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
    window_size = config["window_size"]
    frame_bound = (window_size, len(df))

    env = StockTradingEnv(df, config["window_size"], frame_bound, render_mode)

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    agent = DDQN(state_size, action_size, config["agent"])
    agent.load_model(config["agent"]["model_path"])  # Ensure the model path is correct

    total_reward = 0
    total_profit = 0
    observation, info = env.reset()

    while True:
        action = agent.act(observation.flatten())
        next_observation, reward, done, truncated, info = env.step(action)

        total_reward += reward
        total_profit = info.get("total_profit", total_profit)

        observation = next_observation

        if truncated or done:
            break

    print(colored(f"===== Evaluation Summary =====", "cyan"))
    print(colored(f"Total Reward: {total_reward:.2f}", "magenta"))
    print(colored(f"Total Profit: {total_profit:.2f}", "blue"))
    print(colored("==========================================", "cyan"))

    env.close()


def download_data(stock_ticker: str) -> None:
    """
    Download the data for the stock trading environment.

    Parameters
    ----------
    stock_ticker : str
        The stock ticker to train on.
    """
    data = yf.download(stock_ticker, start="2020-01-01", end="2021-01-01")
    data.to_csv(f"stock_trader/data/{stock_ticker}.csv")


def main() -> None:
    """
    The main function for the stock trading trainer.
    """
    args = parse_arguments()
    config = load_config()

    # Load or train the model based on the `train` argument
    if args.train:
        print("Training mode activated. The model will be saved after training.")
        train(config, args.stock_ticker, args.render_mode)
    else:
        print("Loading the trained model for evaluation.")
        evaluate(config, args.stock_ticker, args.render_mode)

    # Uncomment to download data
    # download_data('AAPL')


if __name__ == "__main__":
    main()
