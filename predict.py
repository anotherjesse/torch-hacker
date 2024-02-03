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
import deps


class Output(BaseModel):
    file: Optional[Path]
    text: Optional[str]


base_dir = os.path.dirname(os.path.realpath(__file__))
app_dir = base_dir + "/app"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.inner = None

    def start_inner(self):
        if self.inner:
            self.inner.terminate()
            time.sleep(1)

        self.inner_port = 4998

        os.environ["PORT"] = str(self.inner_port)
        self.inner = subprocess.Popen(
            ["python", "-m", "cog.server.http"],
            cwd=app_dir,
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
        description: str = Input(
            description="describe this code/experiment", default=""
        ),
        pip: str = Input(description="python deps to install", default=None),
        apt: str = Input(description="apt deps to install", default=None),
        code: str = Input(description="Code to run", default=None),
        inputs: str = Input(description="json to parse for inputs", default="{}"),
    ) -> Output:
        changes = False

        if deps.install_apts(apt):
            changes = True

        if deps.install_pips(pip):
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
                rv = requests.get(f"http://localhost:{self.inner_port}/health-check")
                state = rv.json()
                if state["status"] == "READY":
                    break
                print("not ready yet, state:", state)
            except requests.RequestException as e:
                print(f"nothing listening on {self.inner_port}")

            time.sleep(1)

        try:
            print("sending request")
            r = requests.post(
                f"http://localhost:{self.inner_port}/predictions", json={"input": inputs}
            )
            print("sent request")
            rv = r.json()
            if rv["status"] == "succeeded":
                try:
                    fn = write_data_uri_to_file(rv["output"], base_dir + "output")
                    return Output(file=Path(fn))
                except ValueError as e:
                    return Output(text=rv["output"])
            else:
                return rv["error"]
        except requests.RequestException as e:
            # Handle exceptions
            return str(e)


def update_code(code):
    if code is None:
        return False

    dest = app_dir + "/predict.py"
    if os.path.exists(dest) and open(dest).read() == code:
        print("no changes made")
        return False

    with open(dest, "w") as f:
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
