import os
import cv2
import torch
import numpy as np
from plyfile import PlyData, PlyElement

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.gaussian_model import GaussianModel

from prepare_segmentation import (gaussians,
                                  cameras,
                                  pipeline,
                                  background,
                                  sam_features,
                                  model_path,
                                  args,
                                  dataset)
from preview_segmentation import input_point
from seg_functions import (generate_3d_prompts,
                           gaussian_decomp,
                           project_to_2d,
                           self_prompt,
                           mask_inverse,
                           ensemble)

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
    print("Start Segmentation...")
    mask_id = 2

    # generate 3D prompts
    xyz = gaussians.get_xyz
    prompts_3d = generate_3d_prompts(xyz, cameras[0], input_point)

    multiview_masks = []
    sam_masks = []
    for i, view in enumerate(cameras):
        image_name = view.image_name
        render_pkg = render(view, gaussians, pipeline, background)

        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

        # project 3d prompts to 2d prompts
        prompts_2d = project_to_2d(view, prompts_3d)

        # sam prediction
        sam_mask = self_prompt(prompts_2d, sam_features[image_name], mask_id)
        if len(sam_mask.shape) != 2:
            sam_mask = torch.from_numpy(sam_mask).squeeze(-1).to("cuda")
        else:
            sam_mask = torch.from_numpy(sam_mask).to("cuda")
        sam_mask = sam_mask.long()
        sam_masks.append(sam_mask)

        # mask assignment to gaussians
        point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)

        multiview_masks.append(point_mask.unsqueeze(-1))

        # # gaussian decomposition as an intermediate process
        # if args.gd_interval != -1 \
        #                     and i % args.gd_interval == 0:  #
        #     gaussians = gaussian_decomp(gaussians, view, sam_mask, indices_mask)

    # multi-view label ensemble
    _, final_mask = ensemble(multiview_masks, threshold=args.threshold)

    # save before gaussian decomposition
    save_path = os.path.join(model_path, 'point_cloud/iteration_30000/point_cloud_seg.ply')
    save_gs(gaussians, final_mask, save_path)

    # if gaussian decomposition as a post-process module
    for i, view in enumerate(cameras):
        if args.gd_interval != -1 and i % args.gd_interval == 0:
            input_mask = sam_masks[i]
            gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask.to('cuda'))

    # save after gaussian decomposition
    save_gd_path = os.path.join(model_path, 'point_cloud/iteration_30000/point_cloud_seg_gd.ply')
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