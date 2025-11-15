"""Utility helpers: timestamp parsing, basic feature engineering functions."""
from datetime import datetime

def parse_ts(ts_str_or_val):
    if ts_str_or_val is None:
        return datetime.utcnow()
    if isinstance(ts_str_or_val, (int, float)):
        return datetime.utcfromtimestamp(ts_str_or_val)
    try:
        return datetime.fromisoformat(ts_str_or_val)
    except Exception:
        return datetime.utcnow()