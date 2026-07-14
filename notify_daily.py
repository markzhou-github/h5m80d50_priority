import pandas as pd

import requests

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
            ts_code_part, priority_part = line.split(' : 优先级-')
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




csv_path = "signals/signals_h5m80d50_priority_20260710.csv"

match_strings = [
    "1",
    "2",
    "3",
    "4",
    "0",
]

send_csv_matches_to_bark(
    csv_path=csv_path,
    match_strings=match_strings,
    user="mark",
    title="0710",
)