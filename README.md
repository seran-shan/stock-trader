# Stock Trader

This initiative involves creating a simulated stock trading environment using reinforcement learning techniques. The core of this simulation is a Double Deep Q-Network (DDQN) agent, which is implemented to optimize the returns of a stock trading portfolio. To facilitate this simulation, the project leverages the OpenAI Gym framework, within which it establishes a custom stock trading environment.

## Project Structure

- `.github/`: Contains GitHub workflow configurations.
  - `workflows/`: Contains workflow configuration files like `pylint.yml`.
- `.vscode/`: Contains configuration files for Visual Studio Code.
- `stock_trader/`: Main source code directory.
  - `agents/`: Contains the reinforcement learning agents.
    - `ddqn.py`: Implements the Double DQN agent.
    - `utils/`: Utility functions and classes for the agents.
  - `data/`: Directory for data files.
  - `environments/`: Contains the trading environments.
    - `stock_trading_env.py`: Custom stock trading environment.
  - `logs/`: Directory for log files.
  - `models/`: Directory for saved models.
  - `plots/`: Directory for plot files.
    - `losses/`: Directory for loss plots.
    - `q_values/`: Directory for Q-value plots.
  - `main.py`: Main entry point of the application.
- `tests/`: Contains unit tests for the project.
- `pyproject.toml` and `poetry.lock`: Configuration files for dependencies and project setup.
- `config.yaml`: Configuration file for the application.
- `README.md`: This file.

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

3. Run with graphical rendering:

```sh
python3 stock_trader/main.py [--train/--evaluate] --render
```

4. Choose own stock:

```sh
python3 stock_trader/main.py [--train/--evaluate] --stock_ticker AAPL
```

5. Use offline dataset:

```sh
python3 stock_trader/main.py [--train/--evaluate] --offline
```

6. Download own stock:

```sh
python3 stock_trader/main.py --download AAPl
```
