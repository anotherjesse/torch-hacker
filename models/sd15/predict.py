from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
import torch

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipeline = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.pipeline.to(DEVICE)

    def predict(
        self,
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
    ) -> Path:
        image = self.pipeline(prompt).images[0]
        image.save("output.jpg")
        return Path("output.jpg")
