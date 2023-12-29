"""
This is the main file for the stock trading trainer.
"""
from typing import Any
import argparse
import signal
import time
import numpy as np
from termcolor import colored
import yaml

# pylint: disable=import-error
from environments.stock_trading_env import StockTradingEnv
from stock_trader.agents.ddqn import DDQN
from stock_trader.utils import (
    download_data,
    load_data,
    plot_loss,
    plot_q_value_differences,
)


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
        "--render",
        action="store_true",
        help="Render mode for the environment",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model if this flag is set, otherwise load the model",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the data for the stock ticker if this flag is set",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model if this flag is set, otherwise train the model",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use already downloaded data when training",
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


def train(config: dict, stock_ticker: str, render: bool, offline: bool):
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
    df = load_data(config, stock_ticker, train_data=True, offline=offline)
    window_size = config["window_size"]
    frame_bound = (window_size, len(df))
    num_episodes = config["num_episodes"]

    env = StockTradingEnv(
        df, config["window_size"], frame_bound, "human" if render else None
    )

    # Initialize DDQN agent
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DDQN(state_size, action_size, config["agent"])

    def save_model(sig, frame):
        print("Saving model due to interruption...")
        agent.save_model(f'{config["agent"]["model_path"]}interrupted_model.pt')
        exit(0)

    signal.signal(signal.SIGINT, save_model)

    episode_losses = []  # List to store average loss per episode

    for episode in range(num_episodes):
        total_reward = 0
        total_profit = 0
        global_step = 0
        q_values = []
        action_counts = [0] * env.action_space.n
        td_errors = []
        losses = []

        done = False
        observation, info = env.reset()

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

            loss = agent.replay()
            if loss is not None:
                losses.append(loss)

            agent.calculate_q_value_diff(observation_flat)

            # Update the target network
            # if episeode % config["agent"]["update_target_network_frequency"] == 0:
            agent.update_target_network()
            observation = next_observation

            total_reward += reward
            total_profit = info.get("total_profit", total_profit)
            q_values.append(agent.get_q_value(observation_flat))
            action_counts[action] += 1
            td_errors.append(td_error)

            global_step += 1

            if truncated:
                break

        current_epsilon = agent.get_epsilon()
        current_learning_rate = agent.get_learning_rate()
        average_loss = sum(losses) / len(losses) if losses else 0
        episode_losses.append(average_loss)
        average_td_error = sum(td_errors) / len(td_errors) if td_errors else 0

        # Converting list to np array for tensorboard logging
        q_values = np.array(q_values)
        td_errors = np.array(td_errors)
        action_counts = np.array(action_counts)
        average_q_value_diff = np.array(agent.q_value_diff)

        # Log summary for the episode to tensorboard
        agent.log_metrics(
            episode,
            total_reward,
            current_epsilon,
            current_learning_rate,
            q_values,
            action_counts,
            average_td_error,
            average_loss,
            average_q_value_diff,
            global_step,
        )

        if (episode + 1) % config["save_interval"] == 0:
            agent.save_model(f'{config["agent"]["model_path"]}model_{episode + 1}.pt')

        print(colored(f"===== Episode {episode + 1} Summary =====", "cyan"))
        print(colored(f"Total Reward: {total_reward:.2f}", "magenta"))
        print(colored(f"Total Profit: {total_profit:.2f}", "blue"))
        print(colored(f"Average Loss: {average_loss:.2f}", "red"))
        print(colored("=============================\n", "cyan"))

    env.close()

    plot_loss(
        episode_losses,
        config["data"]["losses_plot_path"] + time.strftime("%Y%m%d-%H%M%S") + ".png",
    )
    plot_q_value_differences(
        agent.q_value_diff,
        config["data"]["q_values_plot_path"] + time.strftime("%Y%m%d-%H%M%S") + ".png",
    )


def evaluate(config: dict, stock_ticker: str, render: bool, offline: bool):
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
    df = (
        load_data(config, stock_ticker, offline=offline)
        if stock_ticker
        else load_data(config, offline=offline)
    )
    window_size = config["window_size"]
    frame_bound = (window_size, len(df))

    env = StockTradingEnv(
        df, config["window_size"], frame_bound, "human" if render else None
    )

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    agent = DDQN(state_size, action_size, config["agent"])
    agent.load_model(config["agent"]["model_path"] + config["agent"]["model_name"])

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

    print(colored("===== Evaluation Summary =====", "cyan"))
    print(colored(f"Total Reward: {total_reward:.2f}", "magenta"))
    print(colored(f"Total Profit: {total_profit:.2f}", "blue"))
    print(colored("==============================", "cyan"))

    env.close()


def main() -> None:
    """
    The main function for the stock trading trainer.
    """
    args = parse_arguments()
    config = load_config()

    # Load or train the model based on the `train` argument
    if args.train:
        print("Training mode activated. The model will be saved after training.")
        train(config, args.stock_ticker, args.render, args.offline)
    elif args.evaluate:
        print("Loading the trained model for evaluation.")
        evaluate(config, args.stock_ticker, args.render, args.offline)
    elif args.download:
        print("Downloading data for the stock ticker.")
        download_data(config, args.stock_ticker)
    else:
        raise ValueError(
            "Please specify either the `train`, `evaluate`, or `download` flag."
        )


if __name__ == "__main__":
    main()
