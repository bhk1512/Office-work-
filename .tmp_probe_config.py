from pathlib import Path
from dashboard.config import AppConfig
from dashboard.data_loader import is_parquet_dataset, load_stringing_compiled_raw, load_stringing_daily

cfg = AppConfig()
print('data_path =', Path(cfg.data_path).resolve())
print('allowed_root =', Path(cfg.allowed_data_root).resolve())
print('parquet_dataset?', is_parquet_dataset(cfg.data_path))
print('stringing_sheet =', cfg.stringing_sheet_name)
print('stringing_probe_dirs =', cfg.stringing_parquet_dirs)

raw = load_stringing_compiled_raw(cfg)
print('stringing_compiled_rows =', len(raw.index))

try:
    daily = load_stringing_daily(cfg)
    print('stringing_daily_rows =', len(daily.index))
except Exception as e:
    print('stringing_daily_load_error:', e)
