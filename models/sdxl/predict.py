from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
    ) -> Path:
        image = self.pipeline(prompt).images[0]
        image.save("output.jpg")
        return Path("output.jpg")
