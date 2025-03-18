import os
import sys
import cv2
import torch
import numpy as np
import dill
from PIL import Image
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

from gaussiansplatting.scene import Scene
from gaussiansplatting.arguments import ModelParams, PipelineParams

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.gaussian_model import GaussianModel
from seg_functions import (generate_3d_prompts,
                           gaussian_decomp,
                           project_to_2d,
                           self_prompt,
                           mask_inverse,
                           ensemble,
                           predictor,
                           get_combined_args,
                           DILL_SAVE_PATH)

def save_gs(pc, indices_mask, save_path):
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--job_id", required=True, type=str)

    args = get_combined_args(parser)

    print("Start Segmentation...")
    mask_id = 2

    DILL_SAVE_FILE = os.path.join(DILL_SAVE_PATH, f"{args.job_id}.dill")
    if os.path.exists(DILL_SAVE_FILE):
        with open(DILL_SAVE_FILE, "rb") as f:
            seg_data = dill.load(f)
    else:
        print("ERROR: Segmentation data not found!", file=sys.stderr)
        sys.exit(1)

    threshold = seg_data["threshold"]
    gd_interval = seg_data["gd_interval"]
    sam_features = seg_data["sam_features"]
    input_point = seg_data["input_point"]

    model_path = args.model_path

    dataset = model.extract(args)
    dataset.model_path = args.model_path
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cameras = scene.getTrainCameras()

    dataset.white_background = True
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    xyz = gaussians.get_xyz

    # generate 3D prompts
    prompts_3d = generate_3d_prompts(xyz, cameras[0], input_point)

    predictor.set_image(seg_data["render_images"][0])

    mask_save_path = os.path.join(model_path, 'sam_masks')

    if not os.path.exists(mask_save_path):
        os.mkdir(mask_save_path)

    multiview_masks = []
    sam_masks = []
    sam_mask_images = {}
    for i, view in enumerate(cameras):
        image_name = view.image_name
        render_pkg = render(view, gaussians, pipeline, background)

        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

        # project 3d prompts to 2d prompts
        prompts_2d = project_to_2d(view, prompts_3d)

        # sam prediction
        sam_mask, sam_mask_image = self_prompt(prompts_2d, sam_features[image_name], mask_id, predictor)

        if len(sam_mask.shape) != 2:
            sam_mask = torch.from_numpy(sam_mask).squeeze(-1).to("cuda")
        else:
            sam_mask = torch.from_numpy(sam_mask).to("cuda")
        sam_mask = sam_mask.long()
        sam_masks.append(sam_mask)

        # mask assignment to gaussians
        point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)

        point_mask_np = point_mask.cpu().numpy().astype(np.uint8) * 255
        sam_mask_images[image_name] = point_mask_np

        multiview_masks.append(point_mask.unsqueeze(-1))

        # # gaussian decomposition as an intermediate process
        # if gd_interval != -1 \
        #                     and i % gd_interval == 0:  #
        #     gaussians = gaussian_decomp(gaussians, view, sam_mask, indices_mask)


    # Render SAM-Mask Images
    for image_name, sam_mask_image in sam_mask_images.items():
        if isinstance(sam_mask_image, np.ndarray):
            sam_mask_image = Image.fromarray(sam_mask_image)
            sam_mask_image_save_path = os.path.join(mask_save_path, f"sam_mask_{image_name}.jpg")
            sam_mask_image.save(sam_mask_image_save_path)

            print(f"Saved: {sam_mask_image_save_path}")
        else:
            print("ERROR: Image in sam_mask_images is no NumPy-Array!")

    # multi-view label ensemble
    _, final_mask = ensemble(multiview_masks, threshold=threshold)

    # save before gaussian decomposition
    save_path = os.path.join(model_path, f'point_cloud/iteration_{args.iteration}/point_cloud_seg.ply')
    save_gs(gaussians, final_mask, save_path)

    # if gaussian decomposition as a post-process module
    for i, view in enumerate(cameras):
        if gd_interval != -1 and i % gd_interval == 0:
            input_mask = sam_masks[i]
            gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask.to('cuda'))

    # save after gaussian decomposition
    save_gd_path = os.path.join(model_path, f'point_cloud/iteration_{args.iteration}/point_cloud_seg_gd.ply')
    save_gs(gaussians, final_mask, save_gd_path)

    # render object images
    seg_gaussians = GaussianModel(dataset.sh_degree)
    seg_gaussians.load_ply(save_gd_path)

    obj_save_path = os.path.join(model_path, 'obj_images')

    if not os.path.exists(obj_save_path):
        os.mkdir(obj_save_path)

    for idx in range(len(cameras)):
        image_name = cameras[idx].image_name
        view = cameras[idx]

        render_pkg = render(view, seg_gaussians, pipeline, background)
        # get sam output mask
        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
        render_image = cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(obj_save_path, '{}.jpg'.format(image_name)), render_image)

    torch.cuda.empty_cache()