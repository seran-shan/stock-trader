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

## Run the application:

1. Run a training session:

```sh
python3 stock_trader/main.py --train
```

2. Run an evaluation session:

```sh
python3 stock_trader/main.py --evaluate
```

3. Run with human mode:

```sh
python3 stock_trader/main.py --render_mode human
```

4. Choose own stock:

```sh
python3 stock_trader/main.py --stock_ticker AAPL
```
