import requests
import json
import sys
import yaml
from pathlib import Path
import glob


def run(**kwargs):
    r = requests.post("http://localhost:5000/predictions", json={"input": kwargs})
    r.raise_for_status()
    rv = r.json()

    if rv["status"] == "succeeded":
        print("output:", rv["output"])

    if rv["error"]:
        print("Error:", rv["error"])


def run_model(model_dir):
    model_dir = Path(model_dir)
    cog = yaml.load(open(model_dir / "cog.yaml"), Loader=yaml.Loader)
    print("cog:", cog)

    pip = " ".join(cog['build'].get("python_packages", []))
    print("pip:", pip)

    apt = " ".join(cog['build'].get("system_packages", []))
    print("apt:", apt)

    predictor = cog['predict'].split(':')[0]
    code = open(model_dir / predictor).read()

    sample_dir = model_dir / "samples" / "*.json"
    print(sample_dir)
    for sample in glob.glob(str(sample_dir)):
        print(f"Running {sample}")
        inputs = json.load(open(sample))
        # we parse and re-serialize to ensure that we are sending valid json
        inputs = json.dumps(inputs)
        print("inputs:", inputs)
        output = run(inputs=inputs, code=code, pip=pip, apt=apt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sample.py model_dir")
        sys.exit(1)

    run_model(sys.argv[1])
# run(pip="transformers diffusers accelerate", inputs=json.dumps({"prompt": "whippet"}))
