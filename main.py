import warnings
warnings.filterwarnings("ignore")

import os, json, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def main(args):
    
    # Parse arguments
    if args.train_phase == "vae" or args.train_phase == "ns":
        from trainer.trainer_vae_sd15 import Trainer
        from configs.training_configs_vae import Configs
        configs = Configs()
    
    elif args.train_phase == "unet":
        from trainer.trainer_unet_sd15 import Trainer
        from configs.training_configs_unet import Configs
        configs = Configs()
    
    # Update configs
    if args.exp_name == "exp_debug":
        configs.train_batch_size = 2
    else:
        if args.train_batch_size:
            configs.train_batch_size = args.train_batch_size
    
    if args.train_batch_size:
        configs.train_batch_size = int(args.train_batch_size)
    
    configs.exp_name = args.exp_name
    configs.train_phase = args.train_phase
    configs.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Run train func
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
