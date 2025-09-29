import pandas as pd
from typing import List, Dict, Tuple


def analyze_trades(trade_log: List[Dict], price_series: pd.Series) -> Tuple[pd.DataFrame, Dict]:
    """分析交易日志，计算每笔交易表现及总体统计。
    trade_log : list of dict
        结构示例: [{'step': int, 'type': 'B'|'S', 'price': float, 'shares': int, ...}, ...]
    price_series : pd.Series
        与 step 对齐的价格序列（通常是收盘价）。index 不重要，但长度需 >= 最大 step。

    Returns
    positions_df : pd.DataFrame
        每笔完整的买->卖 交易记录，包含:
        ['buy_step','sell_step','hold_bars','entry_price','exit_price',
         'return','MFE','MAE','capture_ratio','shares_buy','shares_sell']
    summary : dict
        汇总统计，包括交易数、胜率、平均/中位收益、平均持有周期、平均 MFE/MAE、以及捕获率统计。
    """
    if not trade_log:
        return pd.DataFrame(), {}

    tl = pd.DataFrame(trade_log).sort_values('step').reset_index(drop=True)
    positions = []
    buy_stack = []

    for _, row in tl.iterrows():
        if row['type'] == 'B':
            buy_stack.append(row)
        elif row['type'] == 'S' and buy_stack:
            buy = buy_stack.pop(0)
            sell = row
            start, end = int(buy['step']), int(sell['step'])
            if end <= start or start >= len(price_series):
                continue
            end_clamped = min(end, len(price_series) - 1)
            seg = price_series[start:end_clamped + 1]
            entry = buy['price']
            exit_p = sell['price']
            ret = exit_p / entry - 1 if entry else 0
            mfe = seg.max() / entry - 1 if entry else 0
            mae = seg.min() / entry - 1 if entry else 0
            capture_ratio = 0 if mfe == 0 else ret / mfe
            positions.append({
                'buy_step': start,
                'sell_step': end_clamped,
                'hold_bars': end_clamped - start,
                'entry_price': entry,
                'exit_price': exit_p,
                'return': ret,
                'MFE': mfe,
                'MAE': mae,
                'capture_ratio': capture_ratio,
                'shares_buy': buy.get('shares'),
                'shares_sell': sell.get('shares')
            })

    pos_df = pd.DataFrame(positions)
    if pos_df.empty:
        return pos_df, {}

    summary = {
        'trades': len(pos_df),
        'win_rate': float((pos_df['return'] > 0).mean()),
        'avg_return': float(pos_df['return'].mean()),
        'median_return': float(pos_df['return'].median()),
        'avg_hold_bars': float(pos_df['hold_bars'].mean()),
        'avg_MFE': float(pos_df['MFE'].mean()),
        'avg_MAE': float(pos_df['MAE'].mean()),
        'avg_capture_ratio': float(pos_df['capture_ratio'].mean()),
        'median_capture_ratio': float(pos_df['capture_ratio'].median())
    }
    return pos_df, summary
