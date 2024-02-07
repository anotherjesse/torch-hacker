from cog import BasePredictor, Input, Path
import torch
from diffusers import DiffusionPipeline, LCMScheduler
import os

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # load LCM-LoRA
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    def predict(
        self,
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
        seed: int = Input(description="noise seed", default=1337),
        steps: int = Input(description="number of input steps", default=4),
        guidance_scale: float = Input(
            description="strength of LCM prompt", default=1.0
        ),
    ) -> Path:
        generator = torch.manual_seed(seed)

        image = self.pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save("output.jpg")
        return Path("output.jpg")
