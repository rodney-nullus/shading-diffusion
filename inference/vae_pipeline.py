import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(".")

import torch
import torchvision.transforms.functional as tvf

from safetensors.torch import load_model

from models.geometry_diffusion import GeometryDiffusion
from dataloader.celeba_pbr import get_dataloader
from configs.configs_unet import Configs

def vae_pipeline(configs, vae_model, data_loader, mode, num_samples, device):
    
    # Create train data iterator
    data_iter = iter(data_loader)
    
    vc_pred_list = []
    vc_gt_list = []
    normal_pred_list = []
    normal_gt_list = []
    albedo_pred_list = []
    albedo_gt_list = []
    roughness_pred_list = []
    roughness_gt_list = []
    specular_pred_list = []
    specular_gt_list = []
    for i in range(num_samples):
        
        # Load data
        eval_data = next(data_iter)
        
        if mode == "geo-diff":
            # View coordinates normalization
            g_buffer = [
                eval_data["v_coords"], 
                eval_data["normal"],
                eval_data["albedo"],
                eval_data["roughness"],
                eval_data["specular"]
            ]
            
            model_input = torch.cat(g_buffer, dim=-1).to(device).permute(0,3,1,2)
            model_input = tvf.resize(model_input, (configs.resolution, configs.resolution))
        
        latents = vae_model.encode(model_input).latent_dist.sample()
        model_output = vae_model.decode(latents).sample.clamp(0,1)
        
        vc_norm_pred = model_output[0][:3]
        vc_norm_gt = model_input[0][:3]
        normal_pred = model_output[0][3:6]
        normal_gt = model_input[0][3:6]
        albedo_pred = model_output[0][6:9]
        albedo_gt = model_input[0][6:9]
        roughness_pred = model_output[0][9:10]
        roughness_gt = model_input[0][9:10]
        specular_pred = model_output[0][10:11]
        specular_gt = model_input[0][10:11]
        
        vc_pred_list.append(vc_norm_pred)
        vc_gt_list.append(vc_norm_gt)
        normal_pred_list.append(normal_pred)
        normal_gt_list.append(normal_gt)
        albedo_pred_list.append(albedo_pred)
        albedo_gt_list.append(albedo_gt)
        roughness_pred_list.append(roughness_pred)
        roughness_gt_list.append(roughness_gt)
        specular_pred_list.append(specular_pred)
        specular_gt_list.append(specular_gt)
    
    vc_pred_tensor = torch.cat(vc_pred_list, dim=2)
    vc_gt_tensor = torch.cat(vc_gt_list, dim=2)
    vc_tensor = torch.cat([vc_pred_tensor, vc_gt_tensor], dim=1)
    tvf.to_pil_image(vc_tensor).save("inference/results/vc.png")
    normal_pred_tensor = torch.cat(normal_pred_list, dim=2)
    normal_gt_tensor = torch.cat(normal_gt_list, dim=2)
    normal_tensor = torch.cat([normal_pred_tensor, normal_gt_tensor], dim=1)
    tvf.to_pil_image(normal_tensor).save("inference/results/normal.png")
    albedo_pred_tensor = torch.cat(albedo_pred_list, dim=2)
    albedo_gt_tensor = torch.cat(albedo_gt_list, dim=2)
    albedo_tensor = torch.cat([albedo_pred_tensor, albedo_gt_tensor], dim=1)
    tvf.to_pil_image(albedo_tensor).save("inference/results/albedo.png")
    roughness_pred_tensor = torch.cat(roughness_pred_list, dim=2)
    roughness_gt_tensor = torch.cat(roughness_gt_list, dim=2)
    roughness_tensor = torch.cat([roughness_pred_tensor, roughness_gt_tensor], dim=1)
    tvf.to_pil_image(roughness_tensor).save("inference/results/roughness.png")
    specular_pred_tensor = torch.cat(specular_pred_list, dim=2)
    specular_gt_tensor = torch.cat(specular_gt_list, dim=2)
    specular_tensor = torch.cat([specular_pred_tensor, specular_gt_tensor], dim=1)
    tvf.to_pil_image(specular_tensor).save("inference/results/specular.png")

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs = Configs()
    configs.eval_batch_size = 1
    configs.resolution = 256

    _, eval_loader = get_dataloader(configs)

    geo_diff = GeometryDiffusion(configs=configs)
    vae = geo_diff.vae.to(device)
    # model_path = "experiments/exp_03/geo-diff/diffusion_pytorch_model.safetensors"
    # model_path = "experiments/exp_04_256/geo-diff/diffusion_pytorch_model.safetensors"
    # model_path = "experiments/exp_05_256/geo-diff/diffusion_pytorch_model.safetensors"
    model_path = "experiments/exp_06/geo-diff/diffusion_pytorch_model.safetensors"
    load_model(vae, model_path)
    
    with torch.no_grad():
        vae_pipeline(configs, vae_model=vae, data_loader=eval_loader, mode="geo-diff", num_samples=10, device=device)
    