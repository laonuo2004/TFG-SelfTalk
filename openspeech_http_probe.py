import importlib.util
from pathlib import Path
import sys
import requests

# 加载仓库内的 config.py
cfg_path = Path(__file__).resolve().parent / 'python3.7' / 'config.py'
if not cfg_path.exists():
    print('config.py not found at', cfg_path)
    sys.exit(1)

spec = importlib.util.spec_from_file_location('remote_config', str(cfg_path))
remote_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_config)

ws_cfg = getattr(remote_config, 'ws_connect_config', None)
if not ws_cfg:
    print('ws_connect_config not found in config.py')
    sys.exit(1)

url = ws_cfg.get('base_url')
headers = ws_cfg.get('headers') or {}

print('Probing URL:', url)
print('Sending headers (masked values):')
for k, v in headers.items():
    s = str(v)
    if any(x in k.lower() for x in ['key', 'secret', 'token', 'access', 'app_key']):
        print(f'  {k}: {s[:6]}...')
    else:
        print(f'  {k}: {s}')

try:
    # 发起 GET 请求并打印详细返回
    resp = requests.get(url, headers=headers, timeout=10)
    print('\nHTTP status:', resp.status_code)
    print('Response headers:')
    for k, v in resp.headers.items():
        print(' ', k + ':', v)
    print('\nResponse body (first 2000 chars):')
    print(resp.text[:2000])
except Exception as e:
    print('Request error:', repr(e))
    sys.exit(1)
