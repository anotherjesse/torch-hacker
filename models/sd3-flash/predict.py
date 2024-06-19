from cog import BasePredictor, Input, Path

import torch
from diffusers import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    FlashFlowMatchEulerDiscreteScheduler,
)
from peft import PeftModel
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:

        # Load LoRA
        transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        transformer = PeftModel.from_pretrained(transformer, "jasperai/flash-sd3")

        self.transformer = transformer.to("cuda")

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            transformer=self.transformer,
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        self.pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="scheduler",
        )
        self.pipe = self.pipe.to("cuda")

    def predict(
        self,
        seed: int = Input(
            description="seed", default=42
        ),
        prompt: str = Input(
            description="prompt", default="A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors."
        ),
    ) -> Path:
        generator = torch.manual_seed(seed)

        image = self.pipe(prompt, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]
        image.save("output.jpg")
        return Path("output.jpg")
