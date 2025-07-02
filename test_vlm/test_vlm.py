import warnings, os, sys, json
warnings.filterwarnings('ignore')
sys.path.append('.')

from tqdm import tqdm

import torch
import torchvision.transforms.functional as tvf

from test_loader import TEST
from torch.utils.data.dataloader import DataLoader

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def test(cot=False):
    
    # specify the path to the model
    model_path_7b = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path_13b = "llava-hf/llava-v1.6-vicuna-13b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path_7b)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path_7b, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda:0")
    
    # Load dataloader
    test_dataset = TEST()
    test_loader = DataLoader(test_dataset, 
                               batch_size=1, 
                               shuffle=False, 
                               num_workers=1,
                               pin_memory=True)
    
    answer_dict = dict()
    dataset_iter = iter(test_loader)
    progress_bar = tqdm(range(10), ncols=90)
    for step in progress_bar:
        
        # Load data
        data = next(dataset_iter)
        image = tvf.to_pil_image(data['rgb'][0].permute(2,0,1))
        
        if cot:
            prompt_gt = data['prompt_gt']
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", 
                            "text": "what's the gender of person in the image?"
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{prompt_gt[0]}"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                            "text": "what's the possible age of person in the image?"
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{prompt_gt[2]}"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                            "text": "Summarize the facial appearance of person in the image."
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{prompt_gt[3]}"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                            "text": """Basing on above informaions using one stentence
                            to summarize the feacial features of person in the image
                            including the geneder, age, faical appearance description."""
                        },
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", 
                            "text": """Using one stentence to summarize the feacial features \\
                            of person in the image including the following information, \\
                            geneder, possible age, faical appearance description."""
                        },
                    ],
                },
            ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")
        
        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)
        
        answer_dict[data['file_index'][0]] = processor.decode(output[0], skip_special_tokens=True)
    
    # Save json data
    with open('answer.json', 'w', encoding='utf-8') as outfile:
        json_obj = json.dumps(answer_dict, indent=4, ensure_ascii=False)
        outfile.write(json_obj)

if __name__ == '__main__':
    
    with torch.no_grad():
        test(cot=False)
