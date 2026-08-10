import tushare as ts
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json

TUSHARE_TOKEN = '6d0701013b3c0d193064f94c0d71d0a115171bc34462d012ea6c002b'

CACHE_FILE = Path("/tmp/trade_date_cache.json")

def _today_cn():
    """中国时间 today"""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")


def is_trade_date(dt=None) -> bool:
    """
    判断是否为中国A股交易日（带缓存 & 中国时区）
    """

    # ---------- 解析日期 ----------
    if dt is None:
        target = _today_cn()

    elif isinstance(dt, (datetime, date)):
        target = dt.strftime("%Y%m%d")

    else:
        s = str(dt).replace("-", "")
        if len(s) != 8:
            raise ValueError("date must be YYYYMMDD or YYYY-MM-DD")
        target = s

    # ---------- 读取缓存 ----------
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
            if cache.get("date") == target:
                return cache["is_open"]
        except Exception:
            pass

    # ---------- 查询交易日 ----------
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        df = pro.trade_cal(exchange="SSE", start_date=target, end_date=target)

        is_open = False if df.empty else bool(df.iloc[0]["is_open"])

        # 写缓存
        CACHE_FILE.write_text(json.dumps({"date": target, "is_open": is_open}))

        return is_open

    except Exception:
        # 网络失败时宁愿认为是交易日（避免误关机）
        return True


def last_trade_date(dt=None) -> str:
    """
    返回 dt 之前最近的一个交易日（不含 dt 本身）

    Parameters
    ----------
    dt : datetime/date/str/int/None
        None -> today

    Returns
    -------
    str : YYYYMMDD
    """

    # ---------- 解析日期 ----------
    if dt is None:
        target_date = datetime.now().date()
    elif isinstance(dt, (datetime, date)):
        target_date = dt.date() if isinstance(dt, datetime) else dt
    else:
        s = str(dt).replace("-", "")
        target_date = datetime.strptime(s, "%Y%m%d").date()

    # 向前多查一段（防止春节长假）
    start = (target_date - timedelta(days=30)).strftime("%Y%m%d")
    end = (target_date - timedelta(days=1)).strftime("%Y%m%d")

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    cal = pro.trade_cal(
        exchange="SSE",
        start_date=start,
        end_date=end
    )

    # 只保留开市日
    open_days = cal[cal["is_open"] == 1]["cal_date"].sort_values()

    if open_days.empty:
        raise ValueError("No previous trade day found")

    return open_days.iloc[-1]


def normalize_trade_date(dt) -> str:
    """Normalize date-like input to YYYYMMDD."""
    if dt is None:
        raise ValueError("date cannot be None")
    if isinstance(dt, (datetime, date)):
        return dt.strftime("%Y%m%d")
    s = str(dt).replace("-", "").strip()
    if len(s) != 8:
        raise ValueError("date must be YYYYMMDD or YYYY-MM-DD")
    datetime.strptime(s, "%Y%m%d")
    return s


def trade_dates_between(start_date: str, end_date: str) -> list[str]:
    """Return open A-share trading days between start_date and end_date, inclusive."""
    start = normalize_trade_date(start_date)
    end = normalize_trade_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start} cannot be after end_date {end}")

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    if cal.empty:
        return []
    return cal[cal["is_open"] == 1]["cal_date"].astype(str).sort_values().tolist()

def trade_date_before(end_date: str, trade_days: int = 20) -> str:
    """
    返回 end_date 之前第 trade_days 个交易日（不含 end_date 本身）
    """

    if trade_days < 1:
        raise ValueError("trade_days must be >= 1")

    target = str(end_date).replace("-", "")
    target_date = datetime.strptime(target, "%Y%m%d").date()

    # 20 个交易日通常约 1 个自然月，这里多查一段以覆盖长假。
    start = (target_date - timedelta(days=max(90, trade_days * 5))).strftime("%Y%m%d")
    end = (target_date - timedelta(days=1)).strftime("%Y%m%d")

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    cal = pro.trade_cal(
        exchange="SSE",
        start_date=start,
        end_date=end
    )

    open_days = cal[cal["is_open"] == 1]["cal_date"].sort_values(ascending=False)

    if len(open_days) < trade_days:
        raise ValueError(f"Only found {len(open_days)} open trade days before {target}")

    return open_days.iloc[trade_days - 1]


def resolve_upday_window(
    start_upday: str | None = None,
    end_upday: str | None = None,
    lookback_trade_days: int = 10,
) -> tuple[str, str]:
    """
    Resolve the upday download window.

    This window is only for downloading/updating source data. It should not be
    expanded for feature warmup history.
    """
    resolved_end = normalize_trade_date(end_upday) if end_upday else end_date
    if start_upday:
        resolved_start = normalize_trade_date(start_upday)
    else:
        resolved_start = trade_date_before(resolved_end, trade_days=lookback_trade_days)
    if resolved_start > resolved_end:
        raise ValueError(f"start_upday {resolved_start} cannot be after end_upday {resolved_end}")
    return resolved_start, resolved_end


def resolve_upday_feature_source_window(
    start_upday: str | None = None,
    end_upday: str | None = None,
    lookback_trade_days: int = 20,
    warmup_trade_days: int = 120,
) -> tuple[str, str, str]:
    """
    Resolve the feature preparation window.

    source_start_upday is for local data coverage checks and feature warmup only.
    Download scripts should use resolve_upday_window(), not this helper.
    """
    resolved_start, resolved_end = resolve_upday_window(
        start_upday=start_upday,
        end_upday=end_upday,
        lookback_trade_days=lookback_trade_days,
    )
    source_start = trade_date_before(resolved_start, trade_days=warmup_trade_days)
    return source_start, resolved_start, resolved_end


allday_start_date = '20250901'
End_date = last_trade_date()
end_date = End_date
UPDAY_LOOKBACK_TRADE_DAYS = 30
UPDAY_FEATURE_WARMUP_TRADE_DAYS = 120
start_upday, end_upday = resolve_upday_window(lookback_trade_days=UPDAY_LOOKBACK_TRADE_DAYS)
source_start_upday, _, _ = resolve_upday_feature_source_window(
    start_upday=start_upday,
    end_upday=end_upday,
    warmup_trade_days=UPDAY_FEATURE_WARMUP_TRADE_DAYS,
)
upday_start_date = start_upday

# Backward-compatible aliases for older scripts in this project.
Start_date = allday_start_date
Today_str = End_date

history_start_date = allday_start_date
refresh_start_date = upday_start_date

# End_date = '20260420'
End_date_global = _today_cn()
Today_str_global = End_date_global
