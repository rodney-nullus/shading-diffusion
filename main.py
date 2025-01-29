from dataclasses import dataclass

import torch
from trainer import Trainer

@dataclass
class Configs:
    
    # Experiment settings
    exp_name: str = "exp_01"
    image_size: tuple = (128, 128)  # the generated image resolution
    random_seed: int = 0
    
    # Training settings
    train_mode = "vae"  # `ddpm` for DDPM, `vae` for VAE, `Inference` for inference
    train_batch_size = 10
    learning_rate = 1e-4
    lr_warmup_steps = 500
    total_epochs = 50
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    #gradient_accumulation_steps = 1
    
    # Evaluation settings
    eval_batch_size = 10  # how many images to sample during evaluation
    
    # Path settings
    data_dir = "dataset/celeba-pbr"
    fov_file_dir = "dataset/celeba-pbr/pred_fov.json"
    output_dir = f"output/{exp_name}"  # the model name locally and on the HF Hub
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    
    # Rendering settings
    z_near: float = 1.0
    z_far: float = 800.0
    
    # Hugging Face Hub settings
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = None

def main(configs):
    trainer = Trainer(configs)
    trainer.train()

if __name__ == "__main__":
    configs = Configs()
    main(configs=configs)