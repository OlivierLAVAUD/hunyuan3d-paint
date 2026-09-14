# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)

# fork (Hunyuan3D-Paint): the paint UNet keeps two learned text embeddings
# (`learned_text_clip_gen` / `learned_text_clip_ref`) as *parameters*, but its
# pipeline reads them from the pipeline body rather than from inside a forward
# pass: `hunyuanpaint/pipeline.py` builds `prompt_embeds` from
# `unet.learned_text_clip_gen` before the UNet is entered. accelerate's
# sequential CPU offload keeps every parameter on the
# 'meta' device between forwards, so such a read comes back with no storage and
# the `.to(dtype, device)` a few lines later fails - twenty minutes into a
# low-VRAM run, and only in that mode:
#
#     NotImplementedError: Cannot copy out of meta tensor; no data!
#
# Buffers are what the offload keeps resident: with `offload_buffers=False`
# (what diffusers passes to `cpu_offload`) the hook moves them to the execution
# device once and never puts them back on 'meta'. The state_dict names do not
# change, so the checkpoint still loads with `strict=True` - the tensors only
# switch bags.
#
# The class that holds them comes from the *model directory*:
# `model_index.json` maps the UNet to `["modules", "UNet2p5DConditionModel"]`,
# so diffusers loads and instantiates that dir's `unet/modules.py`. The copy
# under `texgen/hunyuanpaint/unet/` is only imported (for the pipeline's
# annotations) and never instantiated - which is why this fork lives here rather
# than there: editing that copy changes nothing at runtime.
_OFFLOAD_UNSAFE_WEIGHTS = ("learned_text_clip_gen", "learned_text_clip_ref")


def _offloaded_modules(pipeline) -> list:
    """Every ``nn.Module`` a diffusers pipeline holds, in offload order.

    A ``DiffusionPipeline`` is *not* an ``nn.Module`` - it keeps its components
    as plain attributes - and ``modules()`` therefore has to start from each of
    them. Guarding on ``isinstance(pipeline, nn.Module)`` is always False here
    and skips the walk silently, which is the one wrong way to write this.
    """
    if isinstance(pipeline, torch.nn.Module):
        roots = [pipeline]
    else:
        roots = [value for value in vars(pipeline).values() if isinstance(value, torch.nn.Module)]
    modules = []
    for root in roots:
        modules.extend(root.modules())
    return modules


def _pin_offload_unsafe_weights(pipeline) -> None:
    """Turn the weights read outside `forward` into buffers, before offloading.

    Walks every sub-module instead of naming a path inside the loaded UNet, so
    it does not assume how deep the embeddings sit - and does nothing for a
    model that already exposes them as buffers, or does not have them at all.
    """
    for module in _offloaded_modules(pipeline):
        for name in _OFFLOAD_UNSAFE_WEIGHTS:
            param = module._parameters.pop(name, None)
            if param is not None:
                module.register_buffer(name, param.detach())


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name):
        # fork (Hunyuan3D-Paint): the device is pilotable so the service
        # can build the pipelines on CPU first (HY3DGEN_TEXGEN_DEVICE=cpu)
        # and let diffusers' sequential CPU offload stream sub-modules to
        # the GPU - the upstream hardcoded 'cuda' moves the whole model
        # onto the card during construction, which OOMs on small GPUs
        # before any offload hook can run.
        self.device = os.environ.get('HY3DGEN_TEXGEN_DEVICE', 'cuda')
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo'):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=["hunyuan3d-delight-v2-0/*"]
                    )
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[f'{subfolder}/*']
                    )
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
            
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        # fork (Hunyuan3D-Paint): sub-module-level offload, the granularity
        # a small GPU needs - enable_model_cpu_offload moves each *whole*
        # model onto the card when its turn comes, which still requires
        # the largest model to fit in one piece.
        #
        # fork (Hunyuan3D-Paint): and one preparation the offload needs - the
        # weights the pipeline reads from outside a forward pass have to stop
        # being parameters (see _OFFLOAD_UNSAFE_WEIGHTS), because this offload
        # parks parameters on 'meta' in between. Delight has no such read, but
        # the call is harmless there and keeps the rule in one place.
        _pin_offload_unsafe_weights(self.models['delight_model'].pipeline)
        _pin_offload_unsafe_weights(self.models['multiview_model'].pipeline)
        self.models['delight_model'].pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image, seed=None):

        # fork (Hunyuan3D-Paint): upstream's __call__ takes no seed, and the
        # multiview pass hardcodes one, so a caller had no way to reproduce (or
        # vary) a texture. Forwarded to the multiview model, whose optional
        # keyword keeps the old behaviour when this is omitted.
        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)
            
        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]

        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info, seed=seed)

        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        return textured_mesh
