import gymnasium as gym
import itertools as it
import yfinance as yf
import pandas as pd
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class RLEngineEnv(gym.Env):
    """
    Enhanced trading environment with:
    - Position management functions
    - 4 actions: CloseAll, Hold, Long, Short
    - Exposure-based position sizing
    - Short selling capability
    """

    def __init__(
        self,
        data_dict,
        feature_list,
        initial_balance=10_000,
        window_size=200,
        exposure=0.1,
    ):
        super(RLEngineEnv, self).__init__()

        self.data_dict = data_dict
        self.feature_list = feature_list
        self.tickers = list(data_dict.keys())
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.exposure = np.clip(exposure, 0.01, 1.0)  # Ensure exposure between 1-100%

        # Add reporting infrastructure
        self.episode_metrics = []
        self.current_episode_returns = []
        self.current_episode_values = []
        self.current_episode_actions = []

        # Action space: 0=CloseAll, 1=Hold, 2=Long, 3=Short
        self.action_space = spaces.Discrete(4)

        # Observation space remains the same
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_list),), dtype=np.float32
        )

        # Internal state variables
        self.df = None
        self.current_step = None
        self.end_step = None
        self.balance = None
        self.shares_held = 0
        self.position_type = None  # 'long', 'short', or None
        self.entry_price = None
        self.prev_portfolio_value = None

    def _pick_random_ticker_and_start(self):
        ticker = np.random.choice(self.tickers)
        self.df = self.data_dict[ticker].reset_index(drop=True)

        if len(self.df) <= self.window_size:
            start_idx = 0
        else:
            max_start = len(self.df) - self.window_size
            start_idx = np.random.randint(0, max_start)

        self.current_step = start_idx
        self.end_step = min(
            start_idx + self.window_size - 1, len(self.df) - 1
        )  # Fixed end_step

    def _process_episode(self):
        """Calculate performance metrics at episode end"""
        if len(self.current_episode_returns) == 0:
            return

        # Calculate returns statistics
        returns = np.array(self.current_episode_returns)
        values = np.array(self.current_episode_values)

        # Basic metrics
        total_return = (values[-1] - self.initial_balance) / self.initial_balance * 100
        cumulative_returns = (values / self.initial_balance - 1) * 100

        # Sharpe Ratio (using daily returns)
        sharpe_ratio = self._calculate_sharpe(returns)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(values)

        # Trade statistics
        num_trades = sum(1 for a in self.current_episode_actions if a != 1)
        win_rate = self._calculate_win_rate(returns)

        # Store metrics
        self.episode_metrics.append(
            {
                "total_return_pct": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown,
                "num_trades": num_trades,
                "win_rate_pct": win_rate,
                "volatility_pct": returns.std() * 100,
            }
        )

    def _calculate_sharpe(self, returns, risk_free_rate=0.0):
        """Annualized Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        daily_sharpe = (returns.mean() - risk_free_rate) / returns.std()
        return daily_sharpe * np.sqrt(252)  # Annualize assuming daily data

    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown in percentage"""
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100

    def _calculate_win_rate(self, returns):
        """Calculate percentage of profitable trades"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns) * 100

    def generate_report(self):
        """Generate text-based performance report"""
        if not self.episode_metrics:
            print("No trading data available!")
            return

        print("\n=== Trading Performance Report ===")
        print(f"Episodes Analyzed: {len(self.episode_metrics)}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")

        # Aggregate metrics
        avg_return = np.mean([m["total_return_pct"] for m in self.episode_metrics])
        avg_sharpe = np.mean([m["sharpe_ratio"] for m in self.episode_metrics])
        avg_drawdown = np.mean([m["max_drawdown_pct"] for m in self.episode_metrics])
        avg_trades = np.mean([m["num_trades"] for m in self.episode_metrics])
        avg_win_rate = np.mean([m["win_rate_pct"] for m in self.episode_metrics])
        avg_volatility = np.mean([m["volatility_pct"] for m in self.episode_metrics])

        print("\nAverage Performance per Episode:")
        print(f"- Total Return: {avg_return:.2f}%")
        print(f"- Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"- Max Drawdown: {avg_drawdown:.2f}%")
        print(f"- Win Rate: {avg_win_rate:.2f}%")
        print(f"- Volatility: {avg_volatility:.2f}%")
        print(f"- Trades per Episode: {avg_trades:.1f}")

        # Risk-adjusted return metrics
        best_episode = max(self.episode_metrics, key=lambda x: x["total_return_pct"])
        worst_episode = min(self.episode_metrics, key=lambda x: x["total_return_pct"])

        print("\nBest Episode:")
        print(f"- Return: {best_episode['total_return_pct']:.2f}%")
        print(f"- Sharpe: {best_episode['sharpe_ratio']:.2f}")
        print(f"- Trades: {best_episode['num_trades']}")

        print("\nWorst Episode:")
        print(f"- Return: {worst_episode['total_return_pct']:.2f}%")
        print(f"- Drawdown: {worst_episode['max_drawdown_pct']:.2f}%")
        print(f"- Trades: {worst_episode['num_trades']}")
        print("=" * 40)

    def _get_info(self):
        current_price = self.df.loc[self.current_step, "Close"]
        current_value = self.balance
        if self.position_type == "long":
            current_value += self.shares_held * current_price
        elif self.position_type == "short":
            current_value += self.shares_held * current_price  # Negative value
        return {"profit": current_value - self.initial_balance}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [row[f] for f in self.feature_list],
            dtype=np.float32,
        )
        return obs

    def _open_position(self, position_type, current_price):
        """Open new position with exposure-based sizing"""
        if self.position_type is not None:
            return  # Can't open new position while another is active

        max_risk_amount = self.balance * self.exposure
        shares = max_risk_amount / current_price

        if position_type == "long":
            if self.balance >= max_risk_amount:
                self.balance -= max_risk_amount
                self.shares_held = shares
                self.position_type = "long"
                self.entry_price = current_price
        elif position_type == "short":
            # Short selling: we get the proceeds immediately
            self.balance += max_risk_amount  # Receive cash from short sale
            self.shares_held = -shares
            self.position_type = "short"
            self.entry_price = current_price

    def _close_position(self, current_price):
        """Close current position and update balance"""
        if self.position_type == "long":
            # Sell long position
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif self.position_type == "short":
            # Buy back short position
            buyback_cost = abs(self.shares_held) * current_price
            if self.balance >= buyback_cost:
                self.balance -= buyback_cost
                self.shares_held = 0

        self.position_type = None
        self.entry_price = None

    def step(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        executed_action = "hold"

        # Action handling
        if action == 0:  # Close all positions
            if self.position_type is not None:
                self._close_position(current_price)
                executed_action = "close_all"

        elif action == 2:  # Long
            if self.position_type == "short":
                self._close_position(current_price)  # Close existing short first
                executed_action = "close_short_open_long"
            self._open_position("long", current_price)
            if self.position_type == "long":
                executed_action = "open_long"

        elif action == 3:  # Short
            if self.position_type == "long":
                self._close_position(current_price)  # Close existing long first
                executed_action = "close_long_open_short"
            self._open_position("short", current_price)
            if self.position_type == "short":
                executed_action = "open_short"

        # Calculate portfolio value
        current_value = self.balance
        if self.position_type == "long":
            current_value += self.shares_held * current_price
        elif self.position_type == "short":
            current_value += self.shares_held * current_price  # Negative shares * price

        reward = current_value - self.prev_portfolio_value
        self.prev_portfolio_value = current_value

        # Update step
        self.current_step += 1
        terminated = False
        truncated = self.current_step > self.end_step

        # Auto-close position at episode end
        if truncated and self.position_type is not None:
            self._close_position(current_price)

        obs = self._get_obs()
        info = {
            "balance": self.balance,
            "position": self.position_type,
            "shares": self.shares_held,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "action": executed_action,
            "portfolio_value": current_value,
        }

        # Track data for reporting
        self.current_episode_returns.append(reward)
        self.current_episode_values.append(current_value)
        self.current_episode_actions.append(action)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Process previous episode data
        if len(self.current_episode_returns) > 0:
            self._process_episode()
            self.render()

        # Reset tracking variables
        self.current_episode_returns = []
        self.current_episode_values = []
        self.current_episode_actions = []

        self._pick_random_ticker_and_start()
        self.balance = self.initial_balance
        self.shares_held = 0
        self.position_type = None
        self.entry_price = None
        self.prev_portfolio_value = self.initial_balance

        # Return observation and info tuple
        return self._get_obs(), self._get_info()

    def render(self, mode="human"):
        print(self.generate_report())


class RLEngine:
    def __init__(self, tickers, initial_balance=10_000, window_size=60, exposure=0.9):
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.exposure = exposure
        self.tickers = tickers
        self.features = []
        self.model = None
        self.env = None
        self.data = {}

        self.__fecth_data_yfinance(tickers)

    def __fecth_data_yfinance(self, tickers):
        for t in self.tickers:
            df = yf.download(t, period="1mo", interval="5m")
            df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            self.data[t] = df
        self.features = list(df.columns)
        # To exclude tickers that failed to download:
        self.tickers = list(self.data.keys())

    def __create_relation_features(self):
        for t in self.tickers:
            for a, b in it.combinations(self.data[t].keys(), 2):
                self.data[t][f"{a}_{b}"] = self.data[t][a] / self.data[t][b]
        # Update list of features:
        import pudb

        pu.db
        self.features = list(self.data[t].columns)

    def learn(self, total_timesteps, rel_feat=True):
        if not self.data:  # data dict is empty
            raise Exception("[ERROR] No data avilable yet")
        if rel_feat:
            self.__create_relation_features()

        self.env = DummyVecEnv(
            [
                lambda: RLEngineEnv(
                    data_dict=self.data,
                    feature_list=self.features,
                    initial_balance=self.initial_balance,
                    window_size=self.window_size,
                    exposure=self.exposure,
                )
            ]
        )

        # TODO: make training prarmeters available outside
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )

        self.model.learn(total_timesteps)


if __name__ == "__main__":

    # TICKERS = ["CC=F", "GC=F", "ZL=F", "SI=F", "ZC=F", "ZR=F", "CL=F", "ALI=F", "KE=F"]
    TICKERS = ["AAPL", "TSLA"]

    mdl = RLEngine(TICKERS)
    mdl.learn(40_000, True)
