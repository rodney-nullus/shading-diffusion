import warnings
warnings.filterwarnings("ignore")

import os, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

from models.geometry_diffusion import GeometryDiffusion
from models.texture_diffusion import TextureDiffusion


from configs.configs_unet import Configs

def main(args):

    # Parse configurations
    configs = Configs()
    configs.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Update configs
    configs.exp_name = args.exp_name
    
    if args.exp_name == "exp_debug":
        configs.train_batch_size = 2
    
    if args.train_batch_size:
        configs.train_batch_size = int(args.train_batch_size)

    if args.train_phase == "vae":
        from trainer.trainer_vae_sd15 import Trainer
        trainer = Trainer(configs=configs)
        trainer.train()
    
    elif args.train_phase == "unet":
        from trainer.trainer_unet_sd15 import Trainer
        trainer = Trainer(configs=configs)
        trainer.train()

if __name__ == "__main__":
    
    # train options
    parser = argparse.ArgumentParser()
    
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--train_phase", type=str)
    parser.add_argument("--train_batch_size", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args=args)
