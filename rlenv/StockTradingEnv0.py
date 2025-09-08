import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

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

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df  # 股票历史数据
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # 奖励范围

        # 动作空间：买入、卖出、持有等操作，action[0]为操作类型，action[1]为操作比例
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 状态空间：包含价格、持仓、资金等19个特征
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float16)

    def _next_observation(self):
        # 获取当前时间步的状态特征，确保数值稳定性
        def safe_normalize(value, max_val, default=0.0):
            """安全的归一化函数，处理NaN和除零情况"""
            if pd.isna(value) or max_val == 0:
                return default
            return np.clip(value / max_val, 0, 1)
        
        # 获取当前行数据
        current_data = self.df.iloc[self.current_step]
        
        obs = np.array([
            safe_normalize(current_data['open'], MAX_SHARE_PRICE),      # 开盘价归一化
            safe_normalize(current_data['high'], MAX_SHARE_PRICE),      # 最高价归一化
            safe_normalize(current_data['low'], MAX_SHARE_PRICE),       # 最低价归一化
            safe_normalize(current_data['close'], MAX_SHARE_PRICE),     # 收盘价归一化
            safe_normalize(current_data['volume'], MAX_VOLUME),         # 成交量归一化
            safe_normalize(current_data['amount'], MAX_AMOUNT),         # 成交额归一化
            safe_normalize(current_data['adjustflag'], 10),             # 复权标志
            safe_normalize(current_data['tradestatus'], 1),             # 交易状态
            safe_normalize(current_data['pctChg'], 100),                # 涨跌幅百分比
            safe_normalize(current_data.get('peTTM', 0), 1e4),          # 市盈率
            safe_normalize(current_data.get('pbMRQ', 0), 100),          # 市净率
            safe_normalize(current_data.get('psTTM', 0), 100),          # 市销率
            safe_normalize(current_data['pctChg'], 1e3),                # 涨跌幅缩放
            safe_normalize(self.balance, MAX_ACCOUNT_BALANCE),          # 当前资金归一化
            safe_normalize(self.max_net_worth, MAX_ACCOUNT_BALANCE),    # 最大净值归一化
            safe_normalize(self.shares_held, MAX_NUM_SHARES),           # 持有股票数量归一化
            safe_normalize(self.cost_basis, MAX_SHARE_PRICE),           # 持仓成本归一化
            safe_normalize(self.total_shares_sold, MAX_NUM_SHARES),     # 卖出股票数量归一化
            safe_normalize(self.total_sales_value, MAX_NUM_SHARES * MAX_SHARE_PRICE), # 卖出总金额归一化
        ], dtype=np.float32)
        
        # 确保观察值中没有NaN或无穷值
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        return obs

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

        if action_type < 1 and self.balance > 0:
            # 买入：用余额的一定比例买入股票
            total_possible = int(self.balance / current_price)  # 最多可以买多少股
            shares_bought = int(total_possible * amount)        # 实际买入股数
            
            if shares_bought > 0:
                prev_cost = self.cost_basis * self.shares_held      # 之前持仓成本
                additional_cost = shares_bought * current_price     # 新买入成本

                self.balance -= additional_cost                    # 扣除买入金额
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)  # 更新持仓成本
                self.shares_held += shares_bought                  # 更新持有股数

        elif action_type < 2 and self.shares_held > 0:
            # 卖出：用持有股票的一定比例卖出
            shares_sold = int(self.shares_held * amount)       # 卖出股数
            
            if shares_sold > 0:
                self.balance += shares_sold * current_price        # 增加卖出金额
                self.shares_held -= shares_sold                    # 更新持有股数
                self.total_shares_sold += shares_sold              # 累计卖出股数
                self.total_sales_value += shares_sold * current_price # 累计卖出金额

        # 更新净资产
        self.net_worth = self.balance + self.shares_held * current_price

        # 记录最大净资产
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 如果没有持仓，持仓成本归零
        if self.shares_held == 0:
            self.cost_basis = 0

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

        # 计算更稳定的奖励函数
        profit_ratio = (self.net_worth - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
        reward = np.tanh(profit_ratio)  # 使用tanh函数限制奖励范围在[-1, 1]

        # 如果净资产为负或过低，给予惩罚并结束回合
        if self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.1:  # 如果净资产低于初始资金的10%
            reward = -1.0
            done = True

        obs = self._next_observation()  # 新状态
        info = {'net_worth': self.net_worth, 'balance': self.balance, 'shares_held': self.shares_held}

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

        # 如果传入新的数据集，则替换
        if new_df:
            self.df = new_df

        # 当前步数归零（也可随机初始化）
        # self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        obs = self._next_observation()  # 返回初始状态
        info = {}
        return obs, info

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
