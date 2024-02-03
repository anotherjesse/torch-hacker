from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.pipe.to(DEVICE)

    def predict(
        self,
        image: Path = Input(description="initial image", default=None),
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
        strength: float = Input(description="initial image strength", default=0.75),
        width: int = Input(description="width", default=512),
        height: int = Input(description="height", default=512),
    ) -> Path:
        image = Image.open(image).convert("RGB")
        image = image.resize((width, height))

        image = self.pipe(prompt=prompt, image=image, strength=strength).images[0]
        image.save("output.jpg")
        return Path("output.jpg")
