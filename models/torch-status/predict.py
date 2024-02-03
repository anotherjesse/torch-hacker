from cog import BasePredictor, Input, Path
import torch
import time


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


class Predictor(BasePredictor):
    def setup(self) -> None:
        time.sleep(5)

    def predict(
        self,
    ) -> str:
        return "torch device: " + str(DEVICE)
    
print("inner")