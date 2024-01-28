from cog import BasePredictor, Input, Path
import torch
import time


class Predictor(BasePredictor):
    def setup(self) -> None:
        time.sleep(5)

    def predict(
        self,
    ) -> Path:
        with open('output.txt', 'w') as f:
            f.write("torch cuda is available: " + str(torch.cuda.is_available()) + "\n")
        
        return Path("output.txt")
    
print("inner")