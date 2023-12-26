'''
This file contains the stock trading environment

The code is inspired by the following repository:
https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading
'''
from enum import Enum
from time import time
from typing import Any
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Actions(Enum):
    '''
    The actions that can be taken in the stock trading environment.

    Attributes
    ----------
    BUY : int
        Buy the stock.
    SELL : int
        Sell the stock.
    '''
    SELL = 0
    BUY = 1

class Positions(Enum):
    '''
    The positions that can be taken in the stock trading environment.

    Attributes
    ----------
    SHORT : int
        Short the stock.
    LONG : int
        Long the stock.
    '''
    SHORT = 0
    LONG = 1

    def opposite(self):
        '''
        Get the opposite position.

        Returns
        -------
        opposite_position : Positions
            The opposite position.
        '''
        return Positions.SHORT if self == Positions.LONG else Positions.LONG

class StockTradingEnv(gym.Env):
    '''
    A stock trading environment for OpenAI gym
    '''
    metadata: dict[str, list] = {'render.modes': ['human'], 'render_fps': 3}

    def __init__(
            self,
            df: pd.DataFrame,
            window_size: int,
            frame_bound: tuple[int, int] = None,
            render_mode: str = None
        ):
        super(StockTradingEnv, self).__init__()

        assert len(df.shape) == 2
        assert len(frame_bound) == 2
        assert window_size >= 0
        assert render_mode in self.metadata['render.modes'] or render_mode is None

        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.render_mode = render_mode

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float32
        )

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        # Initialize episode
        self.start_step = min(window_size, len(self.df) - 1)
        self.current_step: int | None = None
        self.last_action_step: int| None = None
        self.end_step: int = len(self.prices) - 1
        self.position: Positions | None = None
        self.position_history: list[Positions] | None = None
        self.total_reward: float | None = None
        self.total_profit: float | None = None
        self.first_rendering: bool | None = None
        self.history: dict[str, list] | None = None
        self.truncated: bool | None = None

    def _reset_episode(self) -> None:
        '''
        Resets variables at the beginning of each episode.
        '''
        self.current_step = self.start_step
        self.last_action_step = self.current_step - 1
        self.position = Positions.SHORT
        self.position_history = (self.window_size * [None]) + [self.position]
        self.total_reward = 0.0
        self.total_profit = 1.0
        self.first_rendering = True
        self.history = {}
        self.truncated = False

    def reset(self, *_args, **kwargs) -> tuple[np.array, dict[str, float]]:
        '''
        Reset the state of the environment to an initial state.

        Parameters
        ----------
        *_args : Any
            Any arguments passed to the function.
        **kwargs : Any
            Any keyword arguments passed to the function.

        Returns
        -------
        initial_state : np.array
            The initial state of the environment.
        '''
        # Extract specific arguments if they are present
        seed: int | None = kwargs.get('seed') or None
        options: dict[str, Any] | None = kwargs.get('options') or None
        super().reset(seed=seed, options=options)
        self.action_space.seed(int(np.random.uniform(0, seed if seed is not None else 1)))
        self._reset_episode()

        observation: np.array = self.get_observation()
        info = self.get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: int) -> tuple[np.array, float, bool, bool, dict[str, float]]:
        '''
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            The action taken by the agent.

        Returns
        -------
        observation : np.array
            The observation for the current step.
        step_reward : float
            The reward for the current step.
        done : bool
            Whether the episode is done.
        truncated : bool
            Whether the episode was truncated due to reaching the maximum number of steps.
        info : dict[str, float]
            The information of the portfolio for the current step.
        '''
        self.truncated = False
        self.current_step += 1

        if self.current_step == self.end_step:
            self.truncated = True

        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward

        self.update_profit(action)

        buy_action_and_short_position = (
            action == Actions.BUY.value and self.position == Positions.SHORT
        )
        sell_action_and_long_position = (
            action == Actions.SELL.value and self.position == Positions.LONG
        )

        if buy_action_and_short_position or sell_action_and_long_position:
            self.position = self.position.opposite()
            self.last_action_step = self.current_step

        self.position_history.append(self.position)
        observation: np.array = self.get_observation()
        info: dict[str, float] = self.get_info()

        for key,_ in info.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(info[key])

        if self.render_mode == "human":
            self.render(self.render_mode)

        return observation, step_reward, False, self.truncated, info

    def render(self, mode: str | None = None) -> None:
        '''
        Render the environment to the screen.

        Parameters
        ----------
        mode : str
            The mode to render the environment in.
        '''
        if self.render_mode == None:
            return

        def plot_position(position: Positions, step: int) -> None:
            '''
            Plot the position of the portfolio.

            Parameters
            ----------
            position : Positions
                The position of the portfolio.
            step : int
                The current step.
            '''
            color = 'green' if position == Positions.LONG else 'red'
            plt.scatter(step, self.prices[step], color=color, marker='^')

        start_time = time()
        if self.first_rendering:
            self.first_rendering = False
            plt.cla()
            plt.plot(self.prices, label='Price')
            start_pos = self.position_history[self.start_step]
            plot_position(start_pos, self.start_step)

        plot_position(self.position, self.current_step)

        plt.title(f'Step: {self.current_step}')
        plt.suptitle(
            f'Total Reward: {self.total_reward:.2f}'
            f'Total Profit: {self.total_profit:.2f}'
        )

        end_time = time()
        rendering_time = end_time - start_time
        pause_time = max(0.01, 1 / self.metadata['render_fps'] - rendering_time)
        assert pause_time >= 0., 'Negative pause time - reduce fps or reduce rendering time'

        plt.pause(pause_time)

    def render_all(self, title: str = None) -> None:
        '''
        Render all the information of the environment to the screen.

        Parameters
        ----------
        title : str
            The title of the plot.
        '''
        window_steps = np.arange(len(self.position_history))
        plt.plot(self.prices, label='Price')

        short_steps = []
        long_steps = []
        for i, step in enumerate(window_steps):
            if self.position_history[i] == Positions.SHORT:
                short_steps.append(step)
            else:
                long_steps.append(step)

        plt.plot(short_steps, self.prices[short_steps], 'ro', label='Short')
        plt.plot(long_steps, self.prices[long_steps], 'go', label='Long')

        if title: 
            plt.title(title)

        plt.suptitle(
            f'Total Reward: {self.total_reward:.2f}'
            f'Total Profit: {self.total_profit:.2f}'
        )

    def close(self):
        '''
        Close the environment.
        '''
        plt.close()

    def save_rendering(self, filepath):
        '''
        Save the rendering of the environment to a file.

        Parameters
        ----------
        filepath : str
            The filepath to save the rendering to.
        '''
        plt.savefig(filepath)

    def pause_rendering(self):
        '''
        Pause the rendering of the environment.
        '''
        plt.show()

    def get_observation(self) -> np.array:
        '''
        Get the observation for the current step.

        Returns
        -------
        observation : np.array
            The observation for the current step.
        '''
        start_index = max(self.current_step - self.window_size + 1, 0)
        end_index = self.current_step + 1

        observation = self.signal_features[start_index:end_index]

        if len(observation) < self.window_size:
            padding = np.zeros((self.window_size - len(observation), self.signal_features.shape[1]))
            observation = np.vstack((padding, observation))

        return observation

    def get_info(self) -> dict[str, float]:
        '''
        Get the information of the portfolio for the current step

        Returns
        -------
        info : dict[str, float]
            The information of the portfolio for the current step
        '''
        return {
            'total_reward': self.total_reward,
            'total_profit': self.total_profit,
            'position': self.position.value,
        }

    def update_profit(self, action: int) -> None:
        '''
        Update the profit of the portfolio.

        Parameters
        ----------
        action : int
            The action taken by the agent.
        '''
        trade_occurred = False
        buy_action_and_short_position = (
            action == Actions.BUY.value and self.position == Positions.SHORT
        )
        sell_action_and_long_position =(
            action == Actions.SELL.value and self.position == Positions.LONG
        )

        if buy_action_and_short_position or sell_action_and_long_position:
            trade_occurred = True

        if trade_occurred or self.truncated:
            current_price = self.prices[self.current_step]
            last_trade_price = self.prices[self.last_action_step]
          
            if self.position == Positions.LONG:
                shares = (self.total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self.total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        '''
        Calculate the maximum possible profit for the environment.

        Returns
        -------
        profit : float
            The maximum possible profit for the environment.
        '''
        current_step: int = self.window_size
        last_trade_step: int = current_step - 1
        profit: int = 1.0

        while current_step < len(self.prices):
            position = None
            if self.prices[current_step] < self.prices[current_step - 1]:
                while (current_step < len(self.prices) and
                    self.prices[current_step] < self.prices[current_step - 1]):
                    current_step += 1
                position = Positions.SHORT
            else:
                while (current_step < len(self.prices) and
                    self.prices[current_step] >= self.prices[current_step - 1]):
                    current_step += 1
                position = Positions.LONG

            if position == Positions.LONG:
                current_price = self.prices[current_step - 1]
                last_trade_price = self.prices[last_trade_step]
                shares = (profit * (1 - self.trade_fee_bid_percent)) / last_trade_price
                profit = (shares * (1 - self.trade_fee_ask_percent)) * current_price
            last_trade_step = current_step - 1

        return profit

    def calculate_reward(self, action):
        '''
        Calculate the reward for the current step.

        Parameters
        ----------
        action : int
            The action taken by the agent.

        Returns
        -------
        step_reward : float
            The reward for the current step.
        '''
        step_reward = 0

        buy_action_and_short_position = (
            action == Actions.BUY.value and self.position == Positions.SHORT
        )
        sell_action_and_long_position =(
            action == Actions.SELL.value and self.position == Positions.LONG
        )

        if buy_action_and_short_position or sell_action_and_long_position:
            current_price = self.prices[self.current_step]
            last_trade_price = self.prices[self.last_action_step]
            price_diff = current_price - last_trade_price

            if self.position == Positions.LONG:
                step_reward += price_diff

        return step_reward

    def _process_data(self):
        prices = self.df['Close'].to_numpy()
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices.astype(np.float32), signal_features.astype(np.float32)
