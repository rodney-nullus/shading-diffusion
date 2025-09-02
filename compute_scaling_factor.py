import torch
from configs.training_configs_vae import Configs
from models.vae_shader import VAE
from torch.utils.data.dataloader import DataLoader
from dataloader.celeba_pbr import CELEBAPBR
from tqdm import tqdm

configs = Configs()
train_dataset = CELEBAPBR(configs, mode="train")
train_loader = DataLoader(train_dataset, 
                          batch_size=10, 
                          shuffle=True, 
                          num_workers=configs.num_workers,
                          pin_memory=True)

device = "cuda"

vae = VAE(configs=configs)
vae_weights = torch.load("experiments/exp_23/vae.pth")
result = vae.load_state_dict(vae_weights)
if not result.missing_keys and not result.unexpected_keys:
    print("All weights are successfully loaded.")

vae = vae.to(device)

train_iter = iter(train_loader)

all_latents = []
progress_bar = tqdm(range(len(train_loader)), ncols=90)
for step in progress_bar:
    train_data = next(train_iter)
    with torch.no_grad():
        # input_list = [
        #     train_data["rgb"],
        #     train_data["depth"], 
        #     train_data["normal"],
        #     train_data["albedo"],
        #     train_data["roughness"],
        #     train_data["specular"],
        #     train_data["mask"]
        # ]
        # model_input = torch.cat(input_list, dim=-1).permute(0,3,1,2).to(device)
        posterior = vae.vae.encode(train_data["rgb"].permute(0,3,1,2).to(device)).latent_dist
        latents = posterior.sample()
    all_latents.append(latents)

latents_cat = torch.cat(all_latents, dim=0)

latents_std = latents_cat.std(dim=[0,2,3])

scaling_factor = 1.0 / latents_std
shifting_factor = latents_cat.mean(dim=[0,2,3])

torch.save(scaling_factor, "scaling_factor_exp_23.pth")
torch.save(shifting_factor, "shifting_factor_exp_23.pth")

print("scaling factor:", scaling_factor)
print("shifting factor:", shifting_factor)
