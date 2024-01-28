from cog import BasePredictor, Input, Path, BaseModel
from typing import Optional
import json
import os
import subprocess
import requests
import atexit
import signal
import time
import re
import base64
import mimetypes

class Output(BaseModel):
    file: Optional[Path]
    text: Optional[str]


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.inner = None

    def start_inner(self):
        if self.inner:
            self.inner.terminate()
            time.sleep(1)

        os.environ["PORT"] = "5001"
        self.inner = subprocess.Popen(
            ["python", "-m", "cog.server.http"],
            cwd="/src/app",
        )
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        signal.signal(signal.SIGINT, self.cleanup)

    def cleanup(self):
        if self.inner:
            print("cleaning up")
            self.inner.terminate()

    def predict(
        self,
        description: str = Input(description="describe this code/experiment", default=""),
        pip: str = Input(description="python deps to install", default=None),
        apt: str = Input(description="apt deps to install", default=None),
        code: str = Input(description="Code to run", default=None),
        inputs: str = Input(description="json to parse for inputs", default="{}"),
    ) -> Output:
        changes = False

        if install_apts(apt):
            changes = True

        if install_pips(pip):
            changes = True

        if update_code(code):
            changes = True

        if changes or self.inner is None:
            print("changes made, restarting server")
            self.start_inner()

        inputs = json.loads(inputs)

        while True:
            if self.inner.poll() is not None:
                # The process has terminated, handle accordingly
                self.inner = None
                return "Internal server is not running."

            try:
                rv = requests.get("http://localhost:5001/health-check")
                state = rv.json()
                if state["status"] == "READY":
                    break
                print("not ready yet, state:", state)
            except requests.RequestException as e:
                print("nothing listening on 5001")

            time.sleep(1)

        try:
            print("sending request")
            r = requests.post(
                "http://localhost:5001/predictions", json={"input": inputs}
            )
            print("sent request")
            rv = r.json()
            if rv["status"] == "succeeded":
                try:
                    fn = write_data_uri_to_file(rv["output"], "/src/output")
                    return Output(file = Path(fn))
                except ValueError as e:
                    return Output(text = rv["output"])
            else:
                return rv["error"]
        except requests.RequestException as e:
            # Handle exceptions
            return str(e)


def install_apts(deps):
    if deps is None:
        return False
    
    deps = deps.strip()
    if len(deps) == 0:
        return False
    
    deps = deps.split(" ")
    
    print("installing apt deps", deps)
    
    subprocess.run(["apt-get", "update"], check=True)
    result = subprocess.run(
        ["apt-get", "install", "-y", *deps], capture_output=True, text=True, check=True
    )
    if "newly installed" in result.stdout or "upgraded" in result.stdout:
        print("Changes made in apt packages.")
        print("pip freeze:")
        subprocess.run(["pip", "freeze"], check=True)
        return True
    else:
        print("No changes made in apt packages.")
        return False


def install_pips(deps):
    if deps is None:
        return False
    
    deps = deps.strip()
    if len(deps) == 0:
        return False
    
    deps = deps.split(" ")

    print("installing pip deps", deps)
    result = subprocess.run(
        ["pip", "install", *deps], capture_output=True, text=True, check=True
    )
    if any(
        keyword in result.stdout
        for keyword in [
            "Successfully installed",
            "Successfully uninstalled",
            "Upgraded",
        ]
    ):
        print("Changes made in pip packages.")
        return True
    else:
        print("No changes made in pip packages.")
        return False


def update_code(code):
    if code is None:
        return False
    
    if (
        os.path.exists("/src/app/predict.py")
        and open("/src/app/predict.py").read() == code
    ):
        print("no changes made")
        return False

    with open("/src/app/predict.py", "w") as f:
        f.write(code)
    print("updated code")
    return True


def write_data_uri_to_file(data_uri, file_path):
    # Extract the MIME type and the base64 data from the data URI
    match = re.match(r"data:([^;]+);base64,(.*)", data_uri)
    if not match:
        raise ValueError("Invalid data URI")

    mime_type, base64_data = match.groups()
    extension = mimetypes.guess_extension(mime_type)

    # Handle edge cases where the extension may not be correctly guessed
    if not extension:
        raise ValueError("Could not determine file extension from MIME type")

    # Decode the base64 data
    binary_data = base64.b64decode(base64_data)

    # Write the data to a file
    with open(f"{file_path}{extension}", "wb") as file:
        file.write(binary_data)

    return f"{file_path}{extension}"


print("outer")
