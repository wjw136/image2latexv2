import sys
import os
sys.path.append(os.path.abspath('./'))#包所在的根目录

import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from image_to_latex.lit_models import LitResNetTransformer


filepath="/home/zzengae/jwwang/final_project/data/formula_images/0_0.png"


lit_model = LitResNetTransformer.load_from_checkpoint("/data/zzengae/jwwang/final_project/test1/epoch=1-val_loss=0.26-val_edit_distance=0.90-val_exact_match=0.67.ckpt")
lit_model.freeze()
transform = ToTensorV2()

image = Image.open(filepath)
image_tensor = transform(image=np.array(image))["image"]  # type: ignore
pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())  # type: ignore
decoded = lit_model.tokenizer.decode(pred[0].tolist())  # type: ignore
decoded_str = " ".join(decoded)
print(decoded_str)

# trainer.validate(val_dataloaders=val_dataloaders)
# ans=lit_model.tokenizer.encode('x+y')
# print(lit_model.tokenizer.decode(ans))
