import os
import pickle
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from rlenv.StockTradingEnv0 import StockTradingEnv, INITIAL_ACCOUNT_BALANCE
from evaluation import analyze_trades, plot_profit_curve, plot_trade_points, save_trade_and_analysis
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
HISTORY_WINDOW = 7
REWARD_WINDOW = 20
REWARD_WEIGHTS = {'ret': 1.0, 'dd': 0.5, 'vol': 0.2}


def stock_trade(stock_file):
    day_profits = []
    print("train-file", stock_file)
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # 数据预处理：处理缺失值和异常值
    numeric_columns = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'turn']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if 'volume' in df.columns:
        df['volume'] = df['volume'].replace(0, df['volume'].median())
    if 'amount' in df.columns:
        df['amount'] = df['amount'].replace(0, df['amount'].median())

    print("shape after preprocessing", df.shape)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(
        df,
        debug=False,
        history_window=HISTORY_WINDOW,
        reward_window=REWARD_WINDOW,
        reward_weights=REWARD_WEIGHTS
    )])
    env.reset()

    # 使用更稳定的PPO参数
    model = PPO("MlpPolicy", env, 
                verbose=1, 
                tensorboard_log='./log',
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
                seed=SEED)
    
    model.learn(total_timesteps=int(1e4)) 
    # 训练 10000 步，batch_size=64, 10000/64=156.25, 大约每个数据点训练156次

    # 尝试使用测试数据，如果不存在则使用训练数据进行评估
    test_file = stock_file.replace('train', 'test')
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file)
        # 对测试数据也进行相同的预处理
        for col in numeric_columns:
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(df_test[col].median())
        if 'volume' in df_test.columns:
            df_test['volume'] = df_test['volume'].replace(0, df_test['volume'].median())
        if 'amount' in df_test.columns:
            df_test['amount'] = df_test['amount'].replace(0, df_test['amount'].median())
    else:
        print(f"Test file not found: {test_file}, using training data for evaluation")
        df_test = df  # 使用训练数据

    print("test-file", test_file, "shape", df_test.shape)

    env = DummyVecEnv([lambda: StockTradingEnv(
        df_test,
        debug=True,
        history_window=HISTORY_WINDOW,
        reward_window=REWARD_WINDOW,
        reward_weights=REWARD_WEIGHTS
    )])
    obs = env.reset()
    
    collected_trades = []
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(action)  # VecEnv: step 返回 (obs, rewards, dones, infos)
        current_env = env.envs[0]
        profit = current_env.net_worth - INITIAL_ACCOUNT_BALANCE
        day_profits.append(profit)
        if 'last_trade' in infos[0]:
            collected_trades.append(infos[0]['last_trade'])
        if i < 30:
            print(f"step={i}, action={action}, balance={current_env.balance:.2f}, shares={current_env.shares_held}")
        if dones[0]:
            break
    if collected_trades:
        trade_log = collected_trades
    else:
        try:
            trade_log = env.env_method('get_trade_log')[0]
        except Exception:
            trade_log = env.get_attr('trade_log')[0]
    return day_profits, trade_log, df_test


def find_file(path, name):
    print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            print("found file", fname)
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))
    day_profits, trade_log, df_test = stock_trade(stock_file)
    print(f"Daily profits data (head): {day_profits[:10]} (total={len(day_profits)})")
    if not day_profits:
        print("Warning: No profit data to plot! Abort visualization.")
        return

    out_dir = './img'
    # 利润曲线
    profit_path = plot_profit_curve(day_profits, stock_code, out_dir)
    print(f"Profit curve saved: {profit_path}")

    # 交易点与分析
    if 'close' in df_test.columns:
        price = df_test['close']
        trade_img, trade_csv = plot_trade_points(price, trade_log, stock_code, out_dir)
        if trade_img:
            print(f"Trade point image saved: {trade_img}")
            print(f"Trade log csv saved: {trade_csv}")
            summary = save_trade_and_analysis(stock_code, trade_log, price, out_dir, analyze_trades)
            if summary:
                print('Summary:', summary)
        else:
            print('No trade points to plot or empty trade log.')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits, trade_log, df_used = stock_trade(stock_file)
                group_result.append((profits, trade_log))
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    test_a_stock_trade('sz.002230.科大讯飞.csv')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)

