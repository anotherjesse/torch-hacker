# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
import time


class Predictor(BasePredictor):
    def setup(self) -> None:
        time.sleep(5)

    def predict(
        self,
    ) -> str:
        """Run a single prediction on the model"""
        return "torch cuda is available: " + str(torch.cuda.is_available()) + "\n"
    
print("inner")