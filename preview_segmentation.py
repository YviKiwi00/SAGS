import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import dill
import sys

from seg_functions import predictor, DILL_SAVE_PATH

input_point = 0

def preview_segmentation(job_id, x, y):
    DILL_SAVE_FILE = os.path.join(DILL_SAVE_PATH, f"{job_id}.dill")

    if os.path.exists(DILL_SAVE_FILE):
        with open(DILL_SAVE_FILE, "rb") as f:
            seg_data = dill.load(f)
    else:
        print("ERROR: Segmentation data not found! Did you run the prepare job first?", file=sys.stderr)
        sys.exit(1)

    input_point = (np.asarray([[x, y]])).astype(np.int32)
    input_label = np.ones(len(input_point))

    predictor.set_image(seg_data["render_images"][0])
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    mask_images_base64 = []
    for i in range(min(3, len(masks))):
        mask = (masks[i] * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask)

        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        mask_images_base64.append(mask_base64)

    return mask_images_base64