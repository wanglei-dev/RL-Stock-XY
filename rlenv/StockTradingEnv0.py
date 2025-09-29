import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 100000

class StockTradingEnv(gym.Env):
    """股票交易环境，适用于强化学习训练"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, debug: bool = False, history_window: int = 10,
                 reward_window: int = 20, reward_weights=None):
        super(StockTradingEnv, self).__init__()

        self.df = df  # 股票历史数据
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # 奖励范围
        self.trade_log = []  # 交易日志
        self.debug = debug
        self.history_window = max(1, int(history_window))
        self.reward_window = max(2, int(reward_window))
        # 奖励权重：正向为累计收益，惩罚为回撤与波动
        self.reward_weights = reward_weights or {
            'ret': 1.0,
            'dd': 0.5,
            'vol': 0.2,
        }

        self._price_features = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
        self._account_features = ['balance', 'shares_held', 'cost_basis', 'net_worth']

        # 动作空间：买入、卖出、持有等操作，action[0]为操作类型，action[1]为操作比例
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 状态空间：包含过去 N 天 K 线特征与账户变化（全部归一化到 [0,1]）
        obs_size = self.history_window * (len(self._price_features) + len(self._account_features))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

    def _safe_normalize(self, value, max_val, default=0.0):
        """安全的归一化函数，处理NaN和除零情况"""
        if max_val == 0 or value is None or pd.isna(value) or np.isinf(value):
            return default
        return float(np.clip(value / max_val, 0.0, 1.0))

    def _price_row_to_features(self, row):
        """将一行价格数据映射为归一化特征。缺失字段时使用 0."""
        return [
            self._safe_normalize(row.get('open', 0), MAX_SHARE_PRICE),
            self._safe_normalize(row.get('high', 0), MAX_SHARE_PRICE),
            self._safe_normalize(row.get('low', 0), MAX_SHARE_PRICE),
            self._safe_normalize(row.get('close', 0), MAX_SHARE_PRICE),
            self._safe_normalize(row.get('volume', 0), MAX_VOLUME),
            self._safe_normalize(row.get('amount', 0), MAX_AMOUNT),
            self._safe_normalize(row.get('pctChg', 0), 100),
        ]

    def _account_snapshot(self):
        return [
            self._safe_normalize(self.balance, MAX_ACCOUNT_BALANCE),
            self._safe_normalize(self.shares_held, MAX_NUM_SHARES),
            self._safe_normalize(self.cost_basis, MAX_SHARE_PRICE),
            self._safe_normalize(self.net_worth, MAX_ACCOUNT_BALANCE),
        ]

    def _next_observation(self):
        """构建包含过去 N 天价格与账户历史的观测，展平成一维向量。"""
        # 价格历史窗口（包含当前步）
        start_idx = max(0, self.current_step - self.history_window + 1)
        end_idx = self.current_step + 1
        window_df = self.df.iloc[start_idx:end_idx]

        price_feats = []
        for _, row in window_df.iterrows():
            price_feats.extend(self._price_row_to_features(row))

        price_feat_len = len(self._price_features)
        need_pad = self.history_window - len(window_df)
        if need_pad > 0:
            price_feats = [0.0] * (need_pad * price_feat_len) + price_feats

        # 账户历史窗口
        account_feats = []
        for snap in self._account_history:
            account_feats.extend(snap)

        obs = np.array(price_feats + account_feats, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return obs

    def _compute_reward(self):
        """使用最近 reward_window 个净值点计算奖励：
        - 累计收益: (NW_t / NW_{t-k}) - 1
        - 最大回撤: max_{i<=j}(1 - NW_j / max_{i..j} NW)
        - 波动: 简单收益的标准差
        奖励 = w_ret*cum_ret - w_dd*max_dd - w_vol*vol，经 tanh 限幅到 [-1,1]
        """
        hist = list(self._nw_history) if hasattr(self, '_nw_history') else []
        if len(hist) < 2:
            return 0.0
        nw0 = hist[0]
        nwt = hist[-1]
        if nw0 <= 0:
            return -1.0
        cum_ret = nwt / nw0 - 1.0
        # 最大回撤
        max_nw = hist[0]
        max_dd = 0.0
        for v in hist:
            max_nw = max(max_nw, v)
            dd = 0.0 if max_nw <= 0 else 1.0 - v / max_nw
            if dd > max_dd:
                max_dd = dd
        # 波动（基于简单收益）
        rets = []
        for i in range(1, len(hist)):
            prev = hist[i-1]
            cur = hist[i]
            if prev > 0:
                rets.append(cur/prev - 1.0)
        vol = float(np.std(rets)) if rets else 0.0
        w = self.reward_weights
        raw = w.get('ret', 1.0) * cum_ret - w.get('dd', 0.0) * max_dd - w.get('vol', 0.0) * vol
        return float(np.tanh(raw))

    def _take_action(self, action):
        # 根据智能体动作执行买入/卖出操作
        # 在当前时间步内，随机选取一个价格作为成交价
        current_data = self.df.iloc[self.current_step]
        
        # 确保价格数据有效
        open_price = current_data['open'] if not pd.isna(current_data['open']) else current_data['close']
        close_price = current_data['close'] if not pd.isna(current_data['close']) else open_price
        
        if open_price <= 0 or close_price <= 0:
            return  # 如果价格无效，跳过这一步
            
        current_price = random.uniform(min(open_price, close_price), max(open_price, close_price))

        action_type = action[0]  # 动作类型：0-买入，1-卖出，2-持有
        amount = np.clip(action[1], 0, 1)  # 确保操作比例在[0,1]范围内
        trade_executed = None  # 用于记录日志

        if action_type < 1 and self.balance > 0:
            # 买入：用余额的一定比例买入股票
            total_possible = int(self.balance / current_price)  # 最多可以买多少股
            raw_shares = total_possible * amount
            shares_bought = int(raw_shares)
            if self.debug:
                print(f"[DEBUG BUY] step={self.current_step} price={current_price:.2f} action_type={action_type:.4f} amount={amount:.4f} total_possible={total_possible} raw_shares={raw_shares:.2f} shares_bought={shares_bought}")
            
            if shares_bought > 0:
                prev_cost = self.cost_basis * self.shares_held      # 之前持仓成本
                additional_cost = shares_bought * current_price     # 新买入成本

                self.balance -= additional_cost                    # 扣除买入金额
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)  # 更新持仓成本
                self.shares_held += shares_bought                  # 更新持有股数
                trade_executed = ('B', shares_bought, current_price)

        elif action_type < 2 and self.shares_held > 0:
            # 卖出：用持有股票的一定比例卖出
            raw_shares = self.shares_held * amount
            shares_sold = int(raw_shares)       # 卖出股数 (向下取整)
            if self.debug:
                print(f"[DEBUG SELL] step={self.current_step} price={current_price:.2f} action_type={action_type:.4f} amount={amount:.4f} held={self.shares_held} raw_shares={raw_shares:.2f} shares_sold={shares_sold}")
            
            if shares_sold > 0:
                self.balance += shares_sold * current_price        # 增加卖出金额
                self.shares_held -= shares_sold                    # 更新持有股数
                self.total_shares_sold += shares_sold              # 累计卖出股数
                self.total_sales_value += shares_sold * current_price # 累计卖出金额
                trade_executed = ('S', shares_sold, current_price)

    # 更新净资产
        self.net_worth = self.balance + self.shares_held * current_price

        # 记录最大净资产
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 如果没有持仓，持仓成本归零
        if self.shares_held == 0:
            self.cost_basis = 0

        # 记录账户历史（用于观测）
        if hasattr(self, '_account_history'):
            self._account_history.append(self._account_snapshot())

        if trade_executed:
            t, shares, price = trade_executed
            self.trade_log.append({
                'step': self.current_step,
                'type': t,
                'shares': shares,
                'price': price,
                'balance': self.balance,
                'net_worth': self.net_worth
            })

    def step(self, action):
        # 执行一步交易（买/卖/持有），并返回新状态、奖励等信息
        self._take_action(action)
        done = False
        truncated = False

        self.current_step += 1  # 时间步前进

        # 如果到达数据末尾，结束回合
        if self.current_step >= len(self.df) - 1:
            done = True
            self.current_step = len(self.df) - 1  # 防止越界

        # 维护净值历史并计算基于窗口的奖励
        if hasattr(self, '_nw_history'):
            self._nw_history.append(self.net_worth)
        reward = self._compute_reward()

        # 如果净资产为负或过低，给予惩罚并结束回合
        if self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.1:  # 如果净资产低于初始资金的10%
            reward = -1.0
            done = True

        obs = self._next_observation()  # 新状态
        info = {'net_worth': self.net_worth, 'balance': self.balance, 'shares_held': self.shares_held}
        if self.trade_log:
            info['last_trade'] = self.trade_log[-1]

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None, new_df=None):
        # 重置环境到初始状态，开始新一轮交易
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.balance = INITIAL_ACCOUNT_BALANCE  # 账户初始资金
        self.net_worth = INITIAL_ACCOUNT_BALANCE # 初始净资产
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE # 最大净资产
        self.shares_held = 0  # 持有股票数量
        self.cost_basis = 0   # 持仓成本
        self.total_shares_sold = 0  # 累计卖出股数
        self.total_sales_value = 0  # 累计卖出金额

        if new_df:
            self.df = new_df

        self.current_step = 0
        self.trade_log = []  # 重置交易日志

        # 初始化账户历史队列，长度为 history_window，并预填充当前初始状态
        self._account_history = deque(maxlen=self.history_window)
        init_snap = self._account_snapshot()
        for _ in range(self.history_window):
            self._account_history.append(init_snap)

        # 初始化净值历史队列（奖励窗口），放入初始净值
        self._nw_history = deque(maxlen=self.reward_window)
        self._nw_history.append(self.net_worth)

        obs = self._next_observation()  # 返回初始状态
        info = {}
        return obs, info

    def get_trade_log(self):
        return self.trade_log

    def render(self, mode='human', close=False):
        # 可视化当前交易状态（打印信息）
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE  # 当前利润
        print('-'*30)
        print(f'当前步数: {self.current_step}')
        print(f'账户余额: {self.balance}')
        print(f'持有股票: {self.shares_held} (累计卖出: {self.total_shares_sold})')
        print(f'持仓成本: {self.cost_basis} (累计卖出金额: {self.total_sales_value})')
        print(f'当前净资产: {self.net_worth} (最大净资产: {self.max_net_worth})')
        print(f'当前利润: {profit}')
        return profit