"""
This file contains utility functions for the stock trader app.
"""
from matplotlib import pyplot as plt
import pandas as pd
import yfinance as yf


def load_data(
    config: dict,
    stock_ticker: str = None,
    train_data: bool | None = None,
    offline: bool | None = None,
) -> pd.DataFrame:
    """
    Load the data for the stock trading environment.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration for the stock trading environment.

    Returns
    -------
    data : pd.DataFrame
        The data for the stock trading environment.
    """
    if offline:
        return pd.read_csv(
            f"{config['data']['processed_path']}{config['data']['file_name']}"
        )

    start_date = (
        config["data"]["train"]["start_date"]
        if train_data
        else config["data"]["test"]["start_date"]
    )
    end_date = (
        config["data"]["train"]["end_date"]
        if train_data
        else config["data"]["test"]["end_date"]
    )
    interval = config["data"]["interval"]
    data = yf.download(
        stock_ticker or config["stock_ticker"],
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
    )
    return data


def download_data(config: dict, stock_ticker: str, train: bool | None = None) -> None:
    """
    Download the data for the stock trading environment.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration for the stock trading environment.
    stock_ticker : str
        The stock ticker to train on.
    """
    start_date = (
        config["data"]["train"]["start_date"]
        if train
        else config["data"]["test"]["start_date"]
    )
    end_date = (
        config["data"]["train"]["end_date"]
        if train
        else config["data"]["test"]["end_date"]
    )
    interval = config["data"]["interval"]
    data = yf.download(stock_ticker, start=start_date, end=end_date, interval=interval)
    data.to_csv(
        f"stock_trader/data/raw/"
        f"{stock_ticker}-{interval}-from-{start_date}-to-{end_date}.csv"
    )


def plot_loss(episode_losses: list, saving_path: str):
    """
    Plot the loss per episode.

    Parameters
    ----------
    episode_losses : list[float]
        The losses per episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(episode_losses, label="Loss per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.savefig(saving_path)
    plt.close()


def plot_q_value_differences(q_value_diff: list, saving_path: str):
    """
    Plot the Q-value difference per step.

    Parameters
    ----------
    q_value_diff : list[float]
        The Q-value difference per step.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(q_value_diff, label="Q-value Difference per Step")
    plt.xlabel("Steps")
    plt.ylabel("Average Q-value Difference")
    plt.title("Q-value Convergence Over Time")
    plt.legend()
    plt.savefig(saving_path)
    plt.close()
