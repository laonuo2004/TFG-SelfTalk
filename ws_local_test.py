import asyncio, websockets

async def test():
    url = 'ws://127.0.0.1:6006/ws'
    try:
        print('Connecting to', url)
        ws = await websockets.connect(url, ping_interval=10)
        print('Connected OK (local)')
        await ws.close()
    except Exception as e:
        print('Local connect error:', repr(e))

if __name__ == '__main__':
    asyncio.run(test())
