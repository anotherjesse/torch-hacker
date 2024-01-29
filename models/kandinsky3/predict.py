from cog import BasePredictor, Input, Path
from diffusers import AutoPipelineForText2Image
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")

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
