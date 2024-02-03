from cog import BasePredictor, Input, Path
from diffusers import PixArtAlphaPipeline
import torch


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
        ).to(DEVICE)

    def predict(
        self,
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
    ) -> Path:
        with torch.no_grad():
            image = self.pipe(prompt).images[0]

        image.save("output.jpg")
        return Path("output.jpg")
