import warnings
warnings.filterwarnings("ignore")

import os, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

from models.geometry_diffusion import GeometryDiffusion
from models.texture_diffusion import TextureDiffusion


from configs.configs import Configs

def main(args):

    # Parse configurations
    shd_configs = Configs()

    if args.train_model == "geo-diff":
        from trainer.trainer_sd_v1_5 import Trainer
        
        # Update configs
        if args.exp_name == "exp_debug":
            shd_configs.train_batch_size = 2
        
        if args.train_batch_size:
            shd_configs.train_batch_size = int(args.train_batch_size)
        
        shd_configs.exp_name = args.exp_name
        shd_configs.train_model = args.train_model
        shd_configs.train_phase = args.train_phase
        
        shd_configs.resume_from_checkpoint = args.resume_from_checkpoint
        
        # Load models
        geo_diff = GeometryDiffusion(shd_configs)
        trainer = Trainer(diff_model=geo_diff, configs=shd_configs)
    
    elif args.train_model == "pbnds":
        from trainer.trainer_pbnds import Trainer
        
        # Prase config file
        with open("configs/configs_pbnds.json", 'r') as f:
            pbnds_configs = json.load(f)
        
        # Update configs
        pbnds_configs["exp_name"] = f"{args.exp_name}/pbnds"
        
        # if weight_path is not None:
        #     print(f"Load pretrained weights from: {weight_path}")
        #     pretrained_weights = torch.load(os.path.join(weight_path, 'NeuralRenderer.pth'))
        #     pretrained_weights = {k: v for k, v in pretrained_weights.items() if "unet" not in k}
        # else:
        #     pretrained_weights = None

        trainer = Trainer(pbnds_configs=pbnds_configs, shd_configs=shd_configs)
    
    # Run train function
    trainer.train()

if __name__ == "__main__":
    
    # train options
    parser = argparse.ArgumentParser()
    
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--train_model", type=str)
    parser.add_argument("--train_phase", type=str)
    parser.add_argument("--train_batch_size", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args=args)
