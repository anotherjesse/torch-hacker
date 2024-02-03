import requests
import json
import sys
import yaml
from pathlib import Path
import glob
import os
import replicate


def run(**kwargs):
    port = os.environ.get("PORT", 5000)
    r = requests.post(f"http://localhost:{port}/predictions", json={"input": kwargs})
    r.raise_for_status()
    rv = r.json()

    if rv["status"] == "succeeded":
        return "output: %s" % rv["output"]

    if rv["error"]:
        return "Error: %s" % rv["error"]


def run_model(model_dir, r8_model_and_version=None):
    model_dir = Path(model_dir)
    cog = yaml.load(open(model_dir / "cog.yaml"), Loader=yaml.Loader)
    print("cog:", cog)

    pip = " ".join(cog['build'].get("python_packages", []))
    print("pip:", pip)

    apt = " ".join(cog['build'].get("system_packages", []))
    print("apt:", apt)

    predictor = cog['predict'].split(':')[0]
    code = open(model_dir / predictor).read()

    if os.path.exists(model_dir / "README.md"):
        description = open(model_dir / "README.md").read()
    else:
        description = str(model_dir)

    sample_dir = model_dir / "samples" / "*.json"
    print(sample_dir)
    for sample in glob.glob(str(sample_dir)):
        print(f"Running {sample}")
        inputs = json.load(open(sample))
        # we parse and re-serialize to ensure that we are sending valid json
        inputs = json.dumps(inputs)
        print("inputs:", inputs)
        if r8_model_and_version is None:
            output = run(inputs=inputs, code=code, pip=pip, apt=apt)
        else:
            output = replicate.run(r8_model_and_version, input={
                "description": description,
                "inputs": inputs,
                "code": code,
                "pip": pip,
                "apt": apt
            })
        print(output)



if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_model(sys.argv[1])
    elif len(sys.argv) == 3:
        run_model(sys.argv[2], r8_model_and_version=sys.argv[1])
    else:
        print("usage: python sample.py (optional:r8_model_version) model_dir")
        sys.exit(1)

# run(pip="transformers diffusers accelerate", inputs=json.dumps({"prompt": "whippet"}))
