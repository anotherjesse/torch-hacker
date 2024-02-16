from cog import BasePredictor, Input, Path
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.prior = StableCascadePriorPipeline.from_pretrained(
            "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
        ).to(DEVICE)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(
            "stabilityai/stable-cascade", torch_dtype=torch.float16
        ).to(DEVICE)

    def predict(
        self,
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
        negative_prompt: str = Input(
            description="negative_prompt", default=None
        ),
        steps_a: int = Input(
            description="steps_a", default=20
        ),
        guidance_a: float = Input(
            description="guidance_a", default=4.0
        ),
        steps_b: int = Input(
            description="steps_b", default=10
        ),
        guidance_b: float = Input(
            description="guidance_b", default=0.0
        ),
        width: int = Input(
            description="width", default=1024
        ),
        height: int = Input(
            description="height", default=1024
        ),
        seed: int = Input(
            description="seed", default=1337
        ),
    ) -> Path:
        torch.manual_seed(seed)
        prior_output = self.prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_a,
            num_images_per_prompt=1,
            num_inference_steps=steps_a
        )
        decoder_output = self.decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_b,
            output_type="pil",
            num_inference_steps=steps_b
        ).images

        image = decoder_output[0]
        image.save("output.jpg")
        return Path("output.jpg")
