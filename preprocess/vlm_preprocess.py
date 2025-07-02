import warnings, os, sys, json
warnings.filterwarnings('ignore')
sys.path.append('.')

from tqdm import tqdm
import numpy as np

import torch
import torchvision.transforms.functional as tvf
from transformers import AutoModelForCausalLM

from dataloader.celeba_pbr import CELEBAPBR
from torch.utils.data.dataloader import DataLoader

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from configs.configs_unet import Configs

def preprocess(configs):
    
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # Load dataloader
    celeba_dataset = CELEBAPBR(configs, mode='all')
    celeba_loader = DataLoader(celeba_dataset, 
                               batch_size=1, 
                               shuffle=False, 
                               num_workers=configs.num_workers,
                               pin_memory=True)
    
    questions = ["请用中文回答图片中的人的性别是什么？请从男性或者女性中选择一个", 
                 "请用中文回答图片中的人的肤色是什么？", 
                 "请用中文回答图片中的人的可能年龄是多少？", 
                 "请用中文概括图片中的人的面部特征是什么？",
                 "请用中文描述图片中的人的皮肤状态怎样？"]
    
    answer_dict = dict()
    dataset_iter = iter(celeba_loader)
    progress_bar = tqdm(range(len(dataset_iter)), ncols=90)
    #progress_bar = tqdm(range(10), ncols=90)
    for step in progress_bar:
        
        # Load data
        data = next(dataset_iter)
        pil_images = [tvf.to_pil_image(data['rgb'][0].permute(2,0,1))]
        
        answers = []
        for question in questions:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n<|ref|>{question}<|/ref|>.",
                    "images": pil_images,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt="你是一个 AI 助手，请始终以中文回复用户的问题，避免使用其他语言。"
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=True,
                use_cache=True,
                temperature=0.4,
                top_p=0.9
            )
            
            answers.append(tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True))
        
        answer_dict[data['file_index'][0]] = answers
    
    # Save json data
    with open(os.path.join(configs.data_dir, 'data_prompt.json'), 'w', encoding='utf-8') as outfile:
        json_obj = json.dumps(answer_dict, indent=4, ensure_ascii=False)
        outfile.write(json_obj)

if __name__ == '__main__':
    
    configs = Configs()
    
    with torch.no_grad():
        preprocess(configs)
