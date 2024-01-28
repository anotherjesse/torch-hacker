import requests

r = requests.post("http://localhost:5000/predictions", json={"input": {"apt": "curl"}})
r.raise_for_status()
rv = r.json()

if rv['status'] == 'succeeded':
    print('output:', rv['output'])

if rv['error']:
    print('Error:', rv['error'])