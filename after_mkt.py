import pandas as pd

import requests

from config_date import End_date

users = {
    "mark": "Eqx2dpLcNQdceTffBdXNuL",
    "tim": "QtuPcWU5CBu4Fd8CEewcpW",
    "lmz": "ZmvgRRVm7Ph672J4CujQKj", 
    "minw": "4JjveStyeFjyMQx3wNacRa",
    "ling": "aCSM6bF4yqMqX88XNWsrZB",
    "minzy": "2YX6JE8tAyY3uTdFqcWPag"
}

def send_to_server(lines: list[str]):
    url = 'https://st.vstartour.com/api/mlscreen/'
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': 'ambot_chat_202607',
    }

    # 解析 lines 列表，构建 API 所需的 stocks 数据
    stocks = []
    for line in lines:
        # 假设 line 的格式为 "ts_code : 优先级-priority"
        # 例如: "000700.SZ : 优先级-3"
        try:
            ts_code_part, priority_part = line.split(',')
            ts_code = ts_code_part.strip()
            # 提取 ts_code 的前6位数字作为股票代码
            stock_code = ts_code[:6]
            priority = int(priority_part.strip())
            stocks.append({'code': stock_code, 'confidence': priority})
        except ValueError:
            # 如果某一行格式不正确，可以选择跳过或抛出异常
            print(f"警告：无法解析行 '{line}'，已跳过。")
            continue

    body = {
        'stocks': stocks,
        # 设置截止日期为当前时间（2026-07-12）之后的3天
        'deadline': '2026-07-10',
    }

    resp = requests.post(url, json=body, headers=headers)
    print(resp.json())


def msg_to_server(lines: str):
    url = 'https://st.vstartour.com/api/mlscreen/'
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': 'ambot_chat_202607',
    }
    
    body = {
        'signals': lines, 
    }

    resp = requests.post(url, json=body, headers=headers)
    print(resp.json())


def notify_daily(user: str, title: str, msg: str) :
    key = users.get(user)

    if not key:
        return f"No user: {user}"

    try:
        alert_url = f"https://api.day.app/{key}/Alert Sound?sound=birdsong"
        url = f"https://api.day.app/{key}/{title}/{msg}"
        requests.get(url, timeout=10)
        send_status = requests.get(alert_url, timeout=10)
        if send_status.ok:
            return f"SUCCESS: {user} sent"
        else:
            return f"FAILED: {user} status={send_status.status_code}"
    except: 
        return


def send_csv_matches_to_bark(
    csv_path: str,
    match_strings,
    user: str,
    title: str,
):
    """
    Read a CSV file and send matching rows to Bark.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    match_strings : list[str] or set[str]
        Values to match against Column A.
    user : str
        Bark user.
    title : str
        Bark notification title.
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert to set for fast lookup
    match_strings = set(match_strings)

    # Get first two columns
    col_a = 'priority'
    col_b = 'ts_code'

    # Collect matched rows
    lines = []

    for _, row in df.iterrows():
        print(row[col_a])
        if row[col_a] >= 0:  # in match_strings:
            lines.append(f"{row[col_b]} : 优先级-{row[col_a]}")

    if not lines:
        print("No matching rows.")
        return

    # Bark message
    msg = "\n".join(lines)

    result = notify_daily(user, title, msg)
    print(result)

    send_to_server(lines)

import requests
import argparse

from config_date import End_date

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trade_date",
    default=End_date,
    help=f"Trade date (default: {End_date})"
)
args = parser.parse_args()

trade_date = args.trade_date

priority_path = "signals_h5priority/signals_h5m80d50_priority_" + trade_date + ".csv"
dual_path= "signals_h2dual/signals_"+ trade_date +".csv"
ensemble_path= "signals_h5ensemble/signals_" + trade_date + ".csv"


match_strings = [
    "1",
    "2",
    "3",
    "4",
    "0",
]

priority_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5, 
    "0": 0,    
}

dual_map = {
    "1": 1,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 3, 
    "0": 0, 
}

ensemble_map = {
    "F220_TOP1_LATE_EARLY_GATE": 1, 
    "F220_TOP1_LATE_EARLY_GATE|FAMILY_ONLY": 1,
    "CONSENSUS_3": 1,
    "CONSENSUS_2": 2,
    "FAMILY_ONLY": 3,
}

def get_signals(sigfile: str, col1: str, col2: str, valmap: dict )-> str:
    df = pd.read_csv(sigfile, usecols=[col1, col2])

    df[col2] = df[col2].astype(str)
#    df = df[df[col2]!= '0']
    df[col2] = df[col2].map(valmap)
    return df.to_csv(index=False, header=False)

priority_signals = get_signals(priority_path, 'ts_code', 'priority', priority_map )
dual_signals = get_signals(dual_path, 'ts_code', 'signal_priority', dual_map )
ensemble_signals = get_signals(ensemble_path, 'ts_code', 'signal_tag', ensemble_map)

print('priority_signals\n', priority_signals)
print('dual_signals', dual_signals)
print('ensemble_signals', ensemble_signals)

#notify_daily(user='mark', title= trade_date +'_H5', msg="股票代码， 优先级\n"+priority_signals) 
#notify_daily(user='mark', title= trade_date +'_H2', msg="股票代码， 优先级\n"+dual_signals) 
#notify_daily(user='mark', title= trade_date +'_H5', msg="股票代码， 优先级\n"+ensemble_signals) 
# notify_daily(user='minzy', title=End_date+'_H5', msg="股票代码， 优先级\n"+priority_signals) 
# notify_daily(user='minzy', title=End_date+'_H2', msg="股票代码， 优先级\n"+dual_signals) 
#notify_daily(user='minw', title=End_date+'_H5', msg="股票代码， 优先级\n"+priority_signals) 
#notify_daily(user='minw', title=End_date+'_H2', msg="股票代码， 优先级\n"+dual_signals) 
#notify_daily(user='lmz', title=End_date+'_H5', msg="股票代码， 优先级\n"+priority_signals) 
#notify_daily(user='lmz', title=End_date+'_H2', msg="股票代码， 优先级\n"+dual_signals) 
# notify_daily(user='ling', title=End_date+'_H5', msg="股票代码， 优先级\n"+priority_signals) 
# notify_daily(user='ling', title=End_date+'_H2', msg="股票代码， 优先级\n"+dual_signals) 
# notify_daily(user='tim', title=End_date+'_H5', msg="股票代码， 优先级\n"+priority_signals)                   
# notify_daily(user='tim', title=End_date+'_H2', msg="股票代码， 优先级\n"+dual_signals) 

priority_rows = [row for row in priority_signals.split('\n') if row]
dual_rows = [row for row in dual_signals.split('\n') if row]

priority_msg = trade_date + ',H5M80D50\n' + priority_signals
dual_msg = trade_date + ',H2M80D50\n' + dual_signals

ensemble_rows = [row for row in ensemble_signals.split('\n') if row]
ensemble_msg = trade_date + ',H5M80D50\n' + ensemble_signals

print('priority:\n', priority_msg)
print('dual:\n', dual_msg)
print('ensemble:\n', ensemble_msg)

send_msg = ''

if priority_signals: 
    send_msg = send_msg + priority_msg
if dual_signals: 
    send_msg = send_msg + dual_msg
if ensemble_signals: 
    send_msg = send_msg + ensemble_msg

# msg_to_server(send_msg)