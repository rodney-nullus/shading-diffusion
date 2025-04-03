from typing import Union
from dataclasses import dataclass

@dataclass
class Configs:
    
    # Experiment settings
    pretrained_model_name_or_path: str = "sd-legacy/stable-diffusion-v1-5"
    resolution: Union[int, set] = 256
    random_seed: int = 0
    load_pretrained_vae: bool = False
    load_pretrained_unet: bool = False
    run_phase: str = "train"                                    # `train`, `inference`
    enable_xformers_memory_efficient_attention: bool = False
    
    # Model settings
    ## Diffusion
    geo_diff_inchns: int = 11
    geo_diff_outchns: int = 11
    
    ## VLM
    max_new_tokens: int = 256
    do_sample: bool = True
    use_cache: bool = True
    
    # Dataloader settings
    num_workers: int = 4
    
    # Training settings
    train_model: str = ""                                       # `geo-diff`, `tex-diff`
    train_phase: str = ""                                       # `vae`, `unet`
    train_batch_size: int = 10
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_warmup_steps: int = 500
    total_train_epochs_vae: int = 10
    total_train_epochs_unet: int = 10
    center_crop: bool = False
    random_flip: bool = False
    mixed_precision: str = "no"                                 # `no` for float32, `fp16` for automatic mixed precision
    gradient_accumulation_steps: int = 1
    resume_from_checkpoint: Union[str, None] = None
    noise_offset: bool = True
    prediction_type: str = "epsilon"                            # `epsilon`, `v_prediction`, `sample`
    snr_gamma: float = 5
    max_grad_norm: float = 1.0
    scale_factor: float = 1.0
    lora_rank: int = 4
    
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    
    use_ema: bool = False
    foreach_ema: bool = False
    gradient_checkpointing: bool = False
    allow_tf32: bool = True
    scale_lr: bool = False
    
    timestep_bias_strategy: str = "none"
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 5
    
    # Evaluation settings
    eval_batch_size: int = 10                                   # how many images to sample during evaluation
    save_model_epochs: int = 1
    validation_prompts: str = ""
    
    # Path settings
    data_dir: str = "dataset/celeba-pbr"
    output_dir: str = "experiments"                             # the model name locally and on the HF Hub
    #overwrite_output_dir = True                                # overwrite the old model when re-running the notebook
    
    # Rendering settings
    z_near: float = 1.0
    z_far: float = 800.0
    
    # Hugging Face Hub settings
    push_to_hub: bool = False                                   # whether to upload the saved model to the HF Hub
    hub_model_id: str = "<your-username>/<my-awesome-model>"    # the name of the repository to create on the HF Hub
    #hub_private_repo = None
