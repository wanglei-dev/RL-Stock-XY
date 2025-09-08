import os
import pickle
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from rlenv.StockTradingEnv0 import StockTradingEnv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def stock_trade(stock_file):
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    # 尝试使用测试数据，如果不存在则使用训练数据进行评估
    test_file = stock_file.replace('train', 'test')
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file)
    else:
        print(f"Test file not found: {test_file}, using training data for evaluation")
        df_test = df  # 使用训练数据

    print("test-file", test_file, "shape", df_test.shape)

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        current_env = env.envs[0]  # 获取第一个环境实例
        profit = current_env.net_worth - 10000  # INITIAL_ACCOUNT_BALANCE = 10000
        day_profits.append(profit)
        
        if done:
            break
    return day_profits


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))

    daily_profits = stock_trade(stock_file)
    print(f"Daily profits data: {daily_profits[:10]}...")  # 打印前10个数据点
    print(f"Total data points: {len(daily_profits)}")
    print(f"Data range: min={min(daily_profits) if daily_profits else 'N/A'}, max={max(daily_profits) if daily_profits else 'N/A'}")
    
    if not daily_profits:
        print("Warning: No profit data to plot!")
        return
        
    fig, ax = plt.subplots()
    ax.plot(daily_profits, marker='o', label=stock_code, ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')
    print(f"Graph saved to ./img/{stock_code}.png")


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    test_a_stock_trade('sh.600036')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)

