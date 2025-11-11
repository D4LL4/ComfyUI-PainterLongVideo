# ComfyUI-PainterLongVideo/nodes.py

import torch
import comfy.utils
import comfy.model_management

class PainterLongVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "previous_video": ("IMAGE",),
                "motion_frames": ("INT", {"default": 5, "min": 1, "max": 20}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "initial_reference_image": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "video/painter"
    # 👇 关键：设置短名称，避免自动换行
    DISPLAY_NAME = "PainterLongVideo"  # ← 这是核心！

    def execute(self, positive, negative, vae, width, height, length, batch_size, previous_video, motion_frames, motion_amplitude=1.15, initial_reference_image=None, clip_vision_output=None):
        device = comfy.model_management.intermediate_device()
        
        # 1. 零初始化 latent
        latent_timesteps = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_timesteps, height // 8, width // 8], device=device)

        # 2. 获取 previous_video 最后一帧作为 start_image
        last_frame = previous_video[-1:].clone()
        last_frame_resized = comfy.utils.common_upscale(
            last_frame.movedim(-1, 1), 
            width, height, 
            "bilinear", "center"
        ).movedim(1, -1)

        # 构建图像序列：首帧真实，其余为 0.5 灰色
        image_seq = torch.ones((length, height, width, last_frame_resized.shape[-1]), 
                               device=last_frame_resized.device, 
                               dtype=last_frame_resized.dtype) * 0.5
        image_seq[0] = last_frame_resized[0]
        concat_latent_image = vae.encode(image_seq[:, :, :, :3])

        # 创建 mask
        mask = torch.ones((1, 1, latent_timesteps, height // 8, width // 8), 
                          device=device, dtype=last_frame_resized.dtype)
        mask[:, :, 0] = 0.0

        # 3. 运动幅度增强（慢动作修复）
        if motion_amplitude > 1.0:
            base_latent = concat_latent_image[:, :, 0:1]
            gray_latent = concat_latent_image[:, :, 1:]
            diff = gray_latent - base_latent
            diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
            diff_centered = diff - diff_mean
            scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
            scaled_latent = torch.clamp(scaled_latent, -6, 6)
            concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)

        # 4. 提取运动参考帧（用于 reference_motion）
        ref_motion = previous_video[-motion_frames:].clone()
        if ref_motion.shape[0] > 73:
            ref_motion = ref_motion[-73:]
        ref_motion_resized = comfy.utils.common_upscale(
            ref_motion.movedim(-1, 1), 
            width, height, 
            "bilinear", "center"
        ).movedim(1, -1)

        if ref_motion_resized.shape[0] < 73:
            gray_fill = torch.ones([73, height, width, 3], 
                                   device=ref_motion_resized.device, 
                                   dtype=ref_motion_resized.dtype) * 0.5
            gray_fill[-ref_motion_resized.shape[0]:] = ref_motion_resized
            ref_motion_resized = gray_fill

        ref_motion_latent = vae.encode(ref_motion_resized[:, :, :, :3])
        ref_motion_latent = ref_motion_latent[:, :, -19:]

        # 5. 构建 reference_latents 列表
        ref_latents = []
        ref_last = vae.encode(last_frame_resized[:, :, :, :3])
        ref_latents.append(ref_last)

        if initial_reference_image is not None:
            init_img = initial_reference_image[:1]
            init_img_resized = comfy.utils.common_upscale(
                init_img.movedim(-1, 1), 
                width, height, 
                "bilinear", "center"
            ).movedim(1, -1)
            init_latent = vae.encode(init_img_resized[:, :, :, :3])
            ref_latents.append(init_latent)

        # 6. 注入 conditioning
        def inject_conditioning(cond, values_dict):
            new_cond = []
            for c_tensor, c_dict in cond:
                new_dict = c_dict.copy()
                new_dict.update(values_dict)
                new_cond.append([c_tensor, new_dict])
            return new_cond

        def append_conditioning(cond, key, value_list):
            new_cond = []
            for c_tensor, c_dict in cond:
                new_dict = c_dict.copy()
                if key in new_dict:
                    new_dict[key] = new_dict[key] + value_list
                else:
                    new_dict[key] = value_list
                new_cond.append([c_tensor, new_dict])
            return new_cond

        shared_values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask,
            "reference_motion": ref_motion_latent
        }
        pos_out = inject_conditioning(positive, shared_values)
        neg_out = inject_conditioning(negative, shared_values)

        pos_out = append_conditioning(pos_out, "reference_latents", ref_latents)
        neg_ref_latents = [torch.zeros_like(r) for r in ref_latents]
        neg_out = append_conditioning(neg_out, "reference_latents", neg_ref_latents)

        if clip_vision_output is not None:
            pos_out = inject_conditioning(pos_out, {"clip_vision_output": clip_vision_output})
            neg_out = inject_conditioning(neg_out, {"clip_vision_output": clip_vision_output})

        latent_out = {"samples": latent}
        return (pos_out, neg_out, latent_out)


NODE_CLASS_MAPPINGS = {
    "PainterLongVideo": PainterLongVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterLongVideo": "PainterLongVideo"  # ← 确保映射一致
}
