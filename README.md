# Stock Trader

This project is a stock trading simulation using reinforcement learning.

## Project Structure

- `stock_trader/`: Main source code directory.
  - `agents/`: Contains the reinforcement learning agents.
    - `ddqn.py`: Implements the Double DQN agent.
    - `utils/`: Utility functions and classes for the agents.
  - `data/`: Directory for data files.
    - `processed/`: Processed data files.
    - `raw/`: Raw data files.
  - `environments/`: Contains the trading environments.
    - `gym_trading_env.py`: Gym-compatible trading environment.
    - `stock_trading_env.py`: Custom stock trading environment.
  - `main.py`: Main entry point of the application.
- `tests/`: Contains unit tests for the project.
- `pyproject.toml` and `poetry.lock`: Configuration files for dependencies and project setup.

## Setup

1. Install the required dependencies:

```sh
poetry install
```

2. Activate the virtual environment:

```sh
poetry shell
```

3. Run the application:

```sh
python3 stock_trader/main.py
```

3.1. Run the application with arguments:

```sh
python3 stock_trader/main.py --render_mode=human --stock_ticker=AAPL
```
