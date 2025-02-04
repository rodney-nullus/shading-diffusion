import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from trainer import Trainer

@dataclass
class Configs:
    
    # Experiment settings
    exp_name: str = "exp_01"
    image_width: int = 128
    image_height: int = 128
    random_seed: int = 0
    
    # Model settings
    geo_diff_inchns: int = 6
    geo_diff_outchns: int = 6
    
    # Training settings
    train_mode: str = "geo-vae"  # `ddpm` for DDPM, `geo-vae` for VAE, `Inference` for inference
    train_batch_size: int = 10
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 500
    total_epochs: int = 50
    save_image_epochs: int = 10
    save_model_epochs: int = 10
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    #gradient_accumulation_steps = 1
    
    # Evaluation settings
    eval_batch_size: int = 10  # how many images to sample during evaluation
    
    # Path settings
    data_dir: str = "dataset/celeba-pbr"
    output_dir: str = "output"  # the model name locally and on the HF Hub
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    
    # Rendering settings
    z_near: float = 1.0
    z_far: float = 800.0
    
    # Hugging Face Hub settings
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_model_id: str = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = None

def main(configs):
    trainer = Trainer(configs)
    trainer.train()

if __name__ == "__main__":
    configs = Configs()
    main(configs=configs)
