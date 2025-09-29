import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    font = fm.FontProperties(fname='font/wqy-microhei.ttc')
except Exception:
    font = None


def plot_profit_curve(day_profits: List[float], stock_code: str, out_dir: str) -> str:
    """绘制每日利润曲线并保存图片。

    Returns 保存的文件路径
    """
    if not day_profits:
        raise ValueError("day_profits is empty, cannot plot profit curve")

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(day_profits, marker='o', label=stock_code, ms=6, alpha=0.7, mfc='orange')
    ax.grid(alpha=0.3)
    ax.set_xlabel('step')
    ax.set_ylabel('profit')
    if font:
        ax.legend(prop=font)
    else:
        ax.legend()
    fig.tight_layout()
    fname = os.path.join(out_dir, f'{stock_code}.png')
    fig.savefig(fname)
    plt.close(fig)
    return fname


def plot_trade_points(price: pd.Series, trade_log: List[Dict], stock_code: str, out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """绘制价格及买卖点散点图，保存图片与交易 CSV。

    Returns (图路径, 交易csv路径)，如果 trade_log 为空返回 (None, None)
    """
    if not trade_log:
        return None, None
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(price.reset_index(drop=True), color='#555', lw=1, label='close')
    buys_x = [t['step'] for t in trade_log if t['type'] == 'B']
    buys_p = [t['price'] for t in trade_log if t['type'] == 'B']
    sells_x = [t['step'] for t in trade_log if t['type'] == 'S']
    sells_p = [t['price'] for t in trade_log if t['type'] == 'S']

    if buys_x:
        ax.scatter(buys_x, buys_p, c='red', marker='^', label='Buy', s=50, edgecolors='k')
    if sells_x:
        ax.scatter(sells_x, sells_p, c='blue', marker='v', label='Sell', s=50, edgecolors='k')

    ax.set_xlabel('step')
    ax.set_ylabel('price')
    ax.set_title(f'{stock_code} 交易记录')
    ax.grid(alpha=0.3)
    if font:
        ax.legend(prop=font)
    else:
        ax.legend()
    fig.tight_layout()

    img_path = os.path.join(out_dir, f'{stock_code}_trades.png')
    fig.savefig(img_path)
    plt.close(fig)

    csv_path = os.path.join(out_dir, f'{stock_code}_trades.csv')
    pd.DataFrame(trade_log).to_csv(csv_path, index=False)
    return img_path, csv_path


def save_trade_and_analysis(stock_code: str, trade_log: List[Dict], price: pd.Series, out_dir: str,
                             analyzer_func):
    """综合保存交易分析结果: 持仓明细、汇总与图表。

    analyzer_func: callable(trade_log, price_series)->(positions_df, summary_dict)
    """
    if not trade_log:
        print('No trades to analyze.')
        return {}
    os.makedirs(out_dir, exist_ok=True)
    pos_df, summary = analyzer_func(trade_log, price)
    if not pos_df.empty:
        pos_path = os.path.join(out_dir, f'{stock_code}_positions.csv')
        pos_df.to_csv(pos_path, index=False)
        summary_path = os.path.join(out_dir, f'{stock_code}_trade_summary.csv')
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f'Positions saved: {pos_path}')
        print(f'Summary saved: {summary_path}')
    else:
        print('No closed positions to analyze.')
    return summary
