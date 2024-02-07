# torch-hacker

exploring how to make models that are fast and easily to use on:

- replicate / cuda
- apple silicon

## status

- [x] outer model that can be used on replicate

### local cog usage

1. run cog to launch outer model

```bash
cog run -p 5000 python -m cog.server.http
```

2. ask outer model to run a prediction on the phi2 llm model

```bash
python samples.py models/phi2
```

### macos usage

1. create a virtual environment with cog installed

```bash
python -m venv .venv
source .venv/bin/activate
pip install cog
```

2. run the outer model

Unfortunately there is already something listening on port 5000, so we need to use a different port.

```bash
PORT=4999 python -m cog.server.http
```

3. use a model

We can run a prediction with the Phi-2 language model using this command:

```
PORT=4999 python sample.py models/phi2
```

### replicate usage

1. set your replicate environment

```bash
export REPLICATE_API_KEY=your-api-key
```

2. run the most recent version of the model on replicate

```bash
python sample.py anotherjesse/torch-hacker:0894e3d48c047d5cd2578375ab3f90d76ce6e693eac84fc829523d8a42a5a491 models/phi2
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