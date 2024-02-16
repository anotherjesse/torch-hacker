from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
from PIL import Image
from RealESRGAN import RealESRGAN
import torch


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


class Upscaler(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {}

        for scale in [2, 4, 8]:
            self.models[scale] = RealESRGAN(DEVICE, scale=scale)
            self.models[scale].load_weights(
                f"weights/RealESRGAN_x{scale}.pth",
            )

    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        upscale: int = Input(
            choices=[2, 4, 8], description="Upscaling factor", default=4
        ),
    ) -> Path:
        model = self.models[upscale]
        image = Image.open(str(image)).convert("RGB")
        sr_image = model.predict(image)
        out_path = "/tmp/out.png"
        sr_image.save(out_path)
        return Path(out_path)
    

class SD15(BasePredictor):
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

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.upscale = Upscaler()
        self.upscale.setup()
        self.sd15 = SD15()
        self.sd15.setup()
    
    def predict(
        self,
        upscale: int = Input(
            choices=[2, 4, 8], description="Upscaling factor", default=4
        ),
        prompt: str = Input(
            description="prompt", default="An image of a squirrel in Picasso style"
        ),
    ) -> Path:
        image = self.sd15.predict(prompt)
        image = self.upscale.predict(image=image, upscale=upscale)
        return image