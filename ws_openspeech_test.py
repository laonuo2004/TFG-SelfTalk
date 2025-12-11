import asyncio
import traceback
import websockets
import sys
import importlib.util
from pathlib import Path


def load_config():
    cfg_path = Path(__file__).resolve().parent / 'python3.7' / 'config.py'
    if not cfg_path.exists():
        print('config.py not found at', cfg_path)
        return None
    spec = importlib.util.spec_from_file_location('remote_config', str(cfg_path))
    remote_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote_config)
    return remote_config


remote_config = load_config()
if not remote_config:
    sys.exit(1)

ws_cfg = getattr(remote_config, 'ws_connect_config', None)
if not ws_cfg:
    print('ws_connect_config not found in config.py')
    sys.exit(1)

url = ws_cfg.get('base_url')
headers = ws_cfg.get('headers') or {}

print('Attempting to connect to', url)
print('Using headers:')
for k, v in headers.items():
    if any(x in k.lower() for x in ['key', 'secret', 'token', 'access', 'app_key']):
        try:
            s = str(v)
            print(f'  {k}:', s[:6] + '...' if len(s) > 6 else '***')
        except Exception:
            print(f'  {k}: ***')
    else:
        print(f'  {k}:', v)


async def test_connect():
    try:
        async with websockets.connect(url, extra_headers=headers, ping_interval=10) as ws:
            print('Connected OK to openspeech (unexpected, likely needs auth)')
            await ws.close()
    except websockets.exceptions.InvalidStatusCode as isc:
        print('Connect failed with InvalidStatusCode:', isc.status_code)
        # try to print response headers if available
        resp_headers = getattr(isc, 'response_headers', None)
        if resp_headers:
            print('Response headers from server:')
            for k, v in resp_headers.items():
                print(' ', k + ':', v)
        traceback.print_exc()
    except Exception as e:
        print('Connect failed:')
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_connect())
