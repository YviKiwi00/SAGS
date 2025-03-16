import torch
import numpy as np
import os
import dill
from PIL import Image
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.scene import Scene
from gaussiansplatting.arguments import ModelParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render

from seg_functions import DILL_SAVE_PATH, RENDER_IMAGE_SAVE_PATH, predictor

def get_combined_args(parser : ArgumentParser):
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--threshold", default=0.7, type=float, help='threshold of label voting')
    parser.add_argument("--gd_interval", default=20, type=int, help='interval of performing gaussian decomposition')

    parser.add_argument("--job_id", required=True, type=str)

    args = get_combined_args(parser)

    job_render_image_save_path = os.path.join(RENDER_IMAGE_SAVE_PATH, args.job_id)

    os.makedirs(DILL_SAVE_PATH, exist_ok=True)
    os.makedirs(job_render_image_save_path, exist_ok=True)

    # 3D gaussians
    dataset = model.extract(args)
    dataset.model_path = args.model_path
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cameras = scene.getTrainCameras()

    dataset.white_background = True
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    xyz = gaussians.get_xyz

    print("Prepocessing: extracting SAM features...")

    sam_features = {}
    render_images = []

    for view in tqdm(cameras):
        image_name = view.image_name
        render_pkg = render(view, gaussians, pipeline, background)

        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

        render_images.append(render_image)

        predictor.set_image(render_image)
        sam_features[image_name] = predictor.features

    if render_images:
        # first_image = render_images[0]
        for index, render_image in enumerate(render_images):
            if isinstance(render_image, np.ndarray):
                image = Image.fromarray(render_image)

                RENDER_IMAGE_SAVE_FILE = os.path.join(f"{job_render_image_save_path}", f"render_image_{index}.png")
                image.save(RENDER_IMAGE_SAVE_FILE)

                print(f"First Render_Image saved: {RENDER_IMAGE_SAVE_FILE}")
            else:
                print("ERROR: First image in render_images is no NumPy-Array!")
    else:
        print("Error: render_images is empty!")

    DILL_SAVE_FILE = os.path.join(DILL_SAVE_PATH, f"{args.job_id}.dill")
    with open(DILL_SAVE_FILE, "wb") as f:
        dill.dump({
            "sam_features": sam_features,
            "threshold": args.threshold,
            "gd_interval": args.gd_interval,
            "render_images": render_images
        }, f)

    print(f"Saved segmentation data to {DILL_SAVE_FILE}")