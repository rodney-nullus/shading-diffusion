import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel

def sample_sd_v1(
        configs,
        condition_embeds,
        pipeline: StableDiffusionPipeline,
        vae: AutoencoderKL,
        device,
        weight_dtype
    ):
    pipeline = pipeline.to(device)
    pipeline.torch_dtype = weight_dtype
    pipeline.set_progress_bar_config(disable=True)

    if configs.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if configs.random_seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(configs.random_seed)

    # Sample test images
    output_list = []
    for i in range(3):
        with torch.autocast("cuda"):
            latent = pipeline(prompt_embeds=condition_embeds, num_inference_steps=20, generator=generator, output_type="latent").images
            output = vae.decode(latent / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        output_list.append(output[0])
    
    output_tensor = torch.cat(output_list, dim=2)
    
    if configs.train_model == "geo-diff":
        depth = output_tensor[:3]
        # Inverse denomralization
        depth = 1 / depth 
        normal = output_tensor[3:]
        mat = torch.cat([depth, normal], dim=1)
    
    return mat
