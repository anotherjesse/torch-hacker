# torch-hacker

exploring how to make models that are fast and easily to use on:

- replicate / cuda
- apple silicon

## status

- [x] outer model that can be used on replicate

### macos usage

1. create a virtual environment with cog installed

```bash
python -m venv .venv
source .venv/bin/activate
pip install cog
```

2. run the outer model

Unfortunately there is already something listening on port 5000, so we need to change the port.

```bash
PORT=4999 python -m cog.server.http
```

3. use a model

We can run a prediction with the Phi-2 language model using this command:

```
PORT=4999 python sample.py models/phi2
```


## todo

- [ ] put the inner model in its own virtualenv? (especially for macos, but perhaps for replicate too)
- [ ] test the brew/apt mapping/installer on macos
- [x] support Path inputs (whisper, llava, etc) -- seems to be kinda working (at least with https based "Paths")
- [ ] how to "compile" an optimized cog on replicate
- [ ] support models that leverage github code bases (example [moondream](https://github.com/vikhyat/moondream))
- [ ] "vibe" tests for input/outputs?
- [ ] "streaming" support
- [ ] support non-torch models

## models

- [ ] blip
- [ ] blip-2
- [ ] codeformer
- [ ] esrgan
- [ ] gfpgan
- [ ] LCM
- [ ] llava
- [ ] moondream
- [ ] photomaker
- [ ] instant-id
- [ ] i2vgen-xl
- [x] phi-2
- [ ] siglip
- [x] SD1.5
- [x] SD1.5 img2img
- [x] SDXL
- [ ] styletts2
- [ ] SVD
- [x] whisper-v3-large