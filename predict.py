from cog import BasePredictor, Input, Path
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
        pip: str = Input(description="python deps to install", default=None),
        apt: str = Input(description="apt deps to install", default=None),
        code: str = Input(description="Code to run", default=None),
        inputs: str = Input(description="json to parse for inputs", default="{}"),
    ) -> Path:
        # Check if the internal server is running
        
        changes = False

        if apt is not None:
            deps = apt.split(" ")
            print("installing apt deps", deps)
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", *deps], check=True)
            changes = True

        if pip is not None:
            deps = pip.split(" ")
            print("installing pip deps", deps)
            subprocess.run(["pip", "install", *deps], check=True)
            changes = True

        if code is not None:
            print("updating code")
            with open("/src/app/predict.py", "w") as f:
                f.write(code)
            changes = True

        if changes or self.inner is None:
            print("changes made, restarting server")
            self.start_inner()

        inputs = json.loads(inputs)

        while True:
            if self.inner.poll() is not None:
                # The process has terminated, handle accordingly
                return "Internal server is not running."

            try:
                rv = requests.get("http://localhost:5001/health-check")
                state = rv.json()
                if state['status'] == 'READY':
                    break
            except requests.RequestException as e:
                print("nothing listening on 5001")

            time.sleep(1)

        try:
            print("sending request")
            r = requests.post("http://localhost:5001/predictions", json={"input": inputs})
            print("sent request")
            rv = r.json()
            if rv['status'] == 'succeeded':
                fn = write_data_uri_to_file(rv['output'], '/src/output')
                return Path(fn)
            else:
                return rv['error']
        except requests.RequestException as e:
            # Handle exceptions
            return str(e)

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
