import sys, os, math, json, logging
sys.path.append('..')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
from accelerate.logging import get_logger

from tqdm import tqdm
# from utils.sampler import Sampler

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from dataloader.celeba_pbr import CELEBAPBR
from dataloader.ffhqpbr import FFHQPBR
from torch.utils.data.dataloader import DataLoader

from diffusers import UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from compel import Compel

from utils.io import load_hdr

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Models
from models.vae_shader import VAE
from models.neural_renderer import NeuralRenderer
from models.classical_renderer import GGXShader, BlinnPhongShader

from eval_configs import Configs

class Tester:
    def __init__(self, test_name, data_loader, configs: Configs, save_matrics=False):
        
        self.logger = get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_name = test_name
        self.configs = configs
        self.save_matrics = save_matrics
        self.resolution = 256
        self.data_loder = data_loader
        
        # Create test folder
        self.test_folder = f"{self.test_name}"
        if not os.path.exists(self.test_folder):
            os.mkdir(self.test_folder)
        
        self.working_folder = f"{self.test_name}/{self.configs.eval_model}"
        if not os.path.exists(self.working_folder):
            os.mkdir(self.working_folder)
        
        self.output_folder = f"{self.working_folder}/output"
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
            os.mkdir(self.output_folder+"/rgb")
            # os.mkdir(self.output_folder+"/depth")
            # os.mkdir(self.output_folder+"/normal")
            # os.mkdir(self.output_folder+"/albedo")
            # os.mkdir(self.output_folder+"/roughness")
            # os.mkdir(self.output_folder+"/specular")
            # os.mkdir(self.output_folder+"/mask")
        
        if self.configs.test_renderer:
            self.output_folder = os.path.join(self.output_folder, f"{self.configs.render_model}")
            if not os.path.exists(self.output_folder):
                os.mkdir(self.output_folder)
        
        # Load VAE and pipeline
        tokenizer = CLIPTokenizer.from_pretrained(
            configs.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            configs.pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )

        self.compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)
        
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            configs.pretrained_model_name_or_path, 
            subfolder="unet")
        # unet_weight = torch.load("experiments/exp_test/unet.pth")``
        # unet.load_state_dict(unet_weight)
        unet.enable_xformers_memory_efficient_attention()
        unet.eval()
        
        self.vae_shader: VAE = VAE(configs=configs)
        # vae_weights = torch.load(self.configs.vae_weight_path)
        # vae_load_result = self.vae_shader.load_state_dict(vae_weights)
        # self.vae_shader.eval().to(self.device)
        
        # if not vae_load_result.missing_keys:
        #     print("All weights of VAE are successfully loaded.")
        
        # Load pipeline
        if configs.eval_type == "text2image":
            
            if configs.eval_model == "sd15":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "sd-legacy/stable-diffusion-v1-5"
                ).to("cuda")
            elif configs.eval_model == "sdxl":
                self.pipeline = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", 
                    torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
                ).to("cuda")
            elif configs.eval_model == "sd35":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    configs.pretrained_model_name_or_path
                ).to("cuda")
            elif configs.eval_model == "flux":
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    configs.pretrained_model_name_or_path
                ).to("cuda")
                
        elif configs.eval_type == "image2image":
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                configs.pretrained_model_name_or_path
            ).to("cuda")
        
        # Disable diffuser pipeline progress bar
        self.pipeline.set_progress_bar_config(disable=True)
        
        # Load renderer
        self.neural_renderer: NeuralRenderer = NeuralRenderer()
        # ns_weight = torch.load(self.configs.ns_weight_path)
        # ns_load_result = self.neural_renderer.load_state_dict(ns_weight)
        # self.neural_renderer.eval().to(self.device)
        
        # if not ns_load_result.missing_keys:
        #     print("All weights of NS are successfully loaded.")
        
        self.ggx_shader = GGXShader()
        self.blinphong_shader = BlinnPhongShader()
        
        # Camera position
        cam_pos = torch.tensor([0., 0., 0.])[None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False).to(self.device)
        
        # Metrics
        self.fid = FrechetInceptionDistance(feature=768, 
                                            normalize=True, 
                                            input_img_size=(3,self.resolution,self.resolution),
                                            compute_on_cpu=True).to(self.device)
        self.kid = KernelInceptionDistance(subset_size=50,normalize=True).to(self.device)
        self.inception = InceptionScore(normalize=True).to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        self.mse = torch.nn.MSELoss()
        
        self.generator = torch.manual_seed(30)
    
    def test(self):
        
        # Initial log
        print("***** Running test *****")
        print(f"Training backbone: {self.configs.pretrained_model_name_or_path}")
        print(f"Num examples = {len(self.data_loder)}")
        
        # Run test
        self.mse_list = []
        self.psnr_list = []
        self.ssim_list = []
        self.lpips_list = []
        self._test()
        
        if self.save_matrics:
            # # Save metric data
            # torch.save(torch.tensor(self.mse_list), 
            #            f"{self.result_path}/{self.test_name}/{self.configs.diffusion_model}/mse.pth")
            # torch.save(torch.tensor(self.psnr_list), 
            #            f"{self.result_path}/{self.test_name}/{self.configs.diffusion_model}/psnr.pth")
            # torch.save(torch.tensor(self.ssim_list), 
            #            f"{self.result_path}/{self.test_name}/{self.configs.diffusion_model}/ssim.pth")
            # torch.save(torch.tensor(self.lpips_list), 
            #            f"{self.result_path}/{self.test_name}/{self.configs.diffusion_model}/lpips.pth")
            
            inception_mean, inception_std = self.inception.compute()
            kid_mean, kid_std = self.kid.compute()
            fid = self.fid.compute()
        
            report = {
                # "mse": sum(self.mse_list)/ len(self.mse_list),
                # "psnr": sum(self.psnr_list) / len(self.psnr_list),
                # "ssim": sum(self.ssim_list) / len(self.ssim_list),
                # "lpips": sum(self.lpips_list) / len(self.lpips_list),
                "IS_mean": inception_mean.item(),
                "IS_std": inception_std.item(),
                "kid_mean": kid_mean.item(),
                "kid_std": kid_std.item(),
                "fid": fid.item()
            }
        
            json_object = json.dumps(report, indent=4)
            if self.configs.mllm_aug:
                report_name = "report_mllm.json"
            else:
                report_name = "report.json"
            
            with open(f"{self.working_folder}/{report_name}", "w") as outfile:
                outfile.write(json_object)
    
    def _test(self):
        
        data_iter = iter(self.data_loder)
        
        pbar = tqdm(range(len(data_iter))[:200], ncols=80)
        for _ in pbar:
            
            data_buffer = next(data_iter)
            
            hdri_list = []
            if self.configs.dataset == "celeba":
                
                rgb_gt = data_buffer["rgb"].permute(0,3,1,2).to(self.device)
                depth_gt = data_buffer["depth"].permute(0,3,1,2).to(self.device)
                normal_gt = data_buffer["normal"].permute(0,3,1,2).to(self.device)
                mask_gt = data_buffer["mask"].permute(0,3,1,2).to(self.device)
                albedo_gt = data_buffer["albedo"].permute(0,3,1,2).to(self.device)
                roughness_gt = data_buffer["roughness"].permute(0,3,1,2).to(self.device)
                specular_gt = data_buffer["specular"].permute(0,3,1,2).to(self.device)
                hdri_list.append(data_buffer["hdri"].to(self.device))
                file_index = data_buffer["file_index"][0]
                prompt_embeds = self.compel(data_buffer["prompt"])
            
            elif self.configs.dataset == "ffhq":
                
                rgb_gt = data_buffer["rgb_gt"].to(self.device)
                rgb_gt = rgb_gt.permute(0,3,1,2)
                mask_gt = data_buffer["mask_gt"].to(self.device)
                
                albedo_gt = data_buffer["albedo_gt"].to(self.device)
                
                normal_gt = data_buffer['normal_gt'].to(self.device)
                pos_in_cam_gt = data_buffer['pos_in_cam_gt'].to(self.device)
                # if relighting is not None:
                #     hdri_gt = load_hdr(f'../relighting/data/hdri/{relighting}.exr', resize=False).to(self.device)[None]
                # else:
                #     hdri_gt = data_buffer['hdri_gt'].to(self.device)
                hdri_list.append(data_buffer["hdri_gt"].to(self.device))
                file_index = data_buffer['file_index'][0]
            
            # for hdri_path in sorted(os.listdir(self.configs.hdri_dir)):
            #     hdri_list.append(load_hdr(f"{self.configs.hdri_dir}/{hdri_path}", resize=False).to(self.device)[None])
            
            if configs.eval_type == "text2image":
            
                # G-Buffer generation
                prompt="A portrait of a young woman, looking to the camera, shot with 85mm lens."
                rgb = self.pipeline(prompt_embeds=prompt_embeds, 
                                    negative_prompt="low quality",
                                    generator=self.generator,
                                    guidance_scale=9.0,
                                    output_type="pt").images[0]
                
                # Compute metrics
                # IS
                # rgb_uint = (rgb[None] * 255).clamp(0,255).to(torch.uint8)
                self.inception.update(rgb[None])
                
                # FID
                self.fid.update(rgb_gt, real=True)
                self.fid.update(rgb[None], real=False)
                
                # KID
                self.kid.update(rgb_gt, real=True)
                self.kid.update(rgb[None], real=False)
                
                # rgb.save(f"{self.output_folder}/rgb/pred_rgb_{file_index}.png")
                
                # latent = latent / self.vae_shader.vae.config.scaling_factor
                
                # geo_output, mat_output, _ = self.vae_shader.vae.decode(latent).sample
                
                # geo_output = geo_output * 0.5 + 0.5
                # mat_output = mat_output * 0.5 + 0.5
                
                # rgb_pred = geo_output.clamp(0,1)
                # depth_pred = mat_output[:,0].unsqueeze(1).clamp(0,1)
                # normal_pred = mat_output[:,1:4].clamp(0,1)
                # mask_pred = mat_output[:,4].unsqueeze(1).clamp(0,1)
                # albedo_pred = mat_output[:,5:8].clamp(0,1)
                # roughness_pred = mat_output[:,8].unsqueeze(1).clamp(0,1)
                # specular_pred = mat_output[:,9].unsqueeze(1).clamp(0,1)
                
                # tvf.to_pil_image(rgb_pred[0]).save(f"{self.output_folder}/rgb/pred_rgb_{file_index}.png")
                # tvf.to_pil_image(depth_pred[0]).save(f"{self.output_folder}/depth/pred_dep_{file_index}.png")
                # tvf.to_pil_image(normal_pred[0]).save(f"{self.output_folder}/normal/pred_nor_{file_index}.png")
                # tvf.to_pil_image(mask_pred[0]).save(f"{self.output_folder}/mask/pred_mask_{file_index}.png")
                # tvf.to_pil_image(albedo_pred[0]).save(f"{self.output_folder}/albedo/pred_albe_{file_index}.png")
                # tvf.to_pil_image(roughness_pred[0]).save(f"{self.output_folder}/roughness/pred_roug_{file_index}.png")
                # tvf.to_pil_image(specular_pred[0]).save(f"{self.output_folder}/specular/pred_spec_{file_index}.png")
            
            elif configs.eval_type == "image2image":
                
                rgb = self.pipeline(prompt="A portrait of a young woman, looking to the camera, shot with 85mm lens.", 
                                    negative_prompt="low quality",
                                    image=(rgb_gt-0.5)/0.5,
                                    generator=self.generator).images[0]
                
                rgb = self.pipeline(prompt="Looking to the camera, long hair.", 
                                    image=(rgb_gt-0.5)/0.5,
                                    generator=self.generator).images[0]
                
                rgb.save(f"{self.output_folder}/rgb/pred_rgb_{file_index}.png")
            
            # Rendering
            # for id, hdri in enumerate(hdri_list):
                
            #     if self.resolution != rgb_pred.shape[2]:
            #         rgb_gt = tvf.resize(rgb_gt, (self.resolution, self.resolution), antialias=True)
            #         depth_pred = tvf.resize(depth_pred, (self.resolution, self.resolution), antialias=True)
            #         normal_pred = tvf.resize(normal_pred, (self.resolution, self.resolution), antialias=True)
            #         albedo_pred = tvf.resize(albedo_pred, (self.resolution, self.resolution), antialias=True)
            #         roughness_pred = tvf.resize(roughness_pred, (self.resolution, self.resolution), antialias=True)
            #         specular_pred = tvf.resize(specular_pred, (self.resolution, self.resolution), antialias=True)
            #         mask_pred = tvf.resize(mask_pred, (self.resolution, self.resolution), antialias=True)
                
            #     denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + \
            #             self.configs.z_near) * mask_pred
                    
            #     fov = torch.tensor(10., device=depth_pred.device)[None].repeat(depth_pred.shape[0])
            #     cam_pos_pred = self._get_cam_coords(denorm_depth.unsqueeze(1), 
            #                                         width=self.resolution, 
            #                                         height=self.resolution, 
            #                                         fov=fov)
                
            #     mask = mask_pred.squeeze(1).bool()
            #     render_buffer = {
            #         "rgb_gt": rgb_gt.permute(0,2,3,1),
            #         "normal_gt": (normal_gt.permute(0,2,3,1) * 2)-1,
            #         "albedo_gt": albedo_pred.permute(0,2,3,1),
            #         "roughness_gt": roughness_pred.permute(0,2,3,1),
            #         "specular_gt": specular_pred.permute(0,2,3,1),
            #         "pos_in_cam_gt": cam_pos_pred,
            #         "hdri_gt": hdri,
            #         "mask": mask
            #     }
                
            #     rgb_shading = torch.ones(1,self.resolution,self.resolution,3,device=self.device)
            #     shading_result, _ = self.neural_renderer(render_buffer=render_buffer, num_light_samples=128, inference=True)
            #     rgb_shading[mask] = shading_result
            #     rgb_shading = rgb_shading.permute(0,3,1,2)
            
            #     tvf.to_pil_image(rgb_shading[0]).save(f"{self.output_folder}/shading_{file_index}_light_{id}.png")
            
            # self._test_FID(pred=rgb_shading, gt=rgb_gt.permute(0,3,1,2))
            # self.mse_list.append(self.mse(rgb_shading, rgb_gt.permute(0,3,1,2)).item())
            # self.psnr_list.append(self.psnr(rgb_shading, rgb_gt.permute(0,3,1,2)).item())
            # self.ssim_list.append(self.ssim(rgb_shading, rgb_gt.permute(0,3,1,2)).item())
            # self.lpips_list.append(self.lpips(rgb_shading, rgb_gt.permute(0,3,1,2)).item())
            
            if self.configs.test_renderer:
                
                # Calculate outbound direction
                out_dirs_gt = self.cam_pos - pos_in_cam_gt[mask_gt].unsqueeze(1)
                out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt[mask_gt].shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt.broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                with torch.no_grad():
                    ggx_pred_rgb = self.ggx_shader.render_equation(shading_input)
                
                rgb_vis = torch.zeros(self.resolution,self.resolution,3).to(self.device)
                rgb_vis[mask_gt[0]] = ggx_pred_rgb
                rgb_vis = rgb_vis.permute(2,0,1)
                
                pred = rgb_vis[None]
                
                mse_list.append(self.mse(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self.psnr(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                
                # Calculate outbound direction
                out_dirs_gt = self.cam_pos - pos_in_cam_gt[mask_gt].unsqueeze(1)
                out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt[mask_gt].shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt.broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                with torch.no_grad():
                    blinnphong_pred_rgb = self.blinphong_shader.render_equation(shading_input, 16)
                
                rgb_vis = torch.zeros(self.resolution,self.resolution,3).to(self.device)
                rgb_vis[mask_gt[0]] = blinnphong_pred_rgb
                rgb_vis = rgb_vis.permute(2,0,1)
                
                pred = rgb_vis[None]
                
                self._test_FID(pred=pred, gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self.mse(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self.psnr(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
    
    def _get_cam_coords(self, depth, width, height, fov):
        fovx = torch.deg2rad(fov)
        fovy = 2 * torch.atan(torch.tan(fovx / 2) / (width / height))
        cam_pos = torch.zeros(depth.shape[0], height, width, 3, device=depth.device)
        Y = 1 - (torch.arange(height, device=depth.device) + 0.5) / height
        Y = Y * 2 - 1
        X = (torch.arange(width, device=depth.device) + 0.5) / width
        X = X * 2 - 1
        Y, X = torch.meshgrid(Y, X, indexing="ij")
        cam_pos[..., 0] = depth.squeeze() * X[None,:,:] * torch.tan(fovx[:,None,None] / 2)
        cam_pos[..., 1] = depth.squeeze() * Y[None,:,:] * torch.tan(fovy[:,None,None] / 2)
        cam_pos[..., 2] = -depth.squeeze()
        return cam_pos

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("test_name", type=str)
    parser.add_argument("--eval_type", type=str, default="text2image")
    parser.add_argument("--eval_model", type=str)
    parser.add_argument("--render_model", type=str, default="ns")
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--mllm", type=bool, default=False)
    parser.add_argument("--save_matrics", type=bool, default=True)
    
    args = parser.parse_args()
    
    configs = Configs()
    
    # Update configs
    configs.eval_type = args.eval_type
    configs.eval_model = args.eval_model
    configs.render_model = args.render_model
    configs.dataset = args.dataset
    configs.mllm = args.mllm
    
    if args.dataset == 'celeba':
        dataset = CELEBAPBR(configs, mode="eval")
    elif args.dataset == 'ffhq':
        dataset = FFHQPBR(data_path="/home/zhuo/remote_nfs/pbnds/data/ffhqpbr_256", 
                          mode='test')
    
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    tester = Tester(test_name=args.test_name, data_loader=test_loader, configs=configs, save_matrics=True)
    
    # Evaluations
    with torch.no_grad():
        tester.test()
