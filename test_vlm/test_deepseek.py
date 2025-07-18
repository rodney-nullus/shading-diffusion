import warnings, os, sys, json
warnings.filterwarnings('ignore')
sys.path.append('.')

from tqdm import tqdm

import torch
import torchvision.transforms.functional as tvf

from test_loader import TEST
from torch.utils.data.dataloader import DataLoader

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from transformers import AutoModelForCausalLM

def test(cot=False):
    
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # Load dataloader
    test_dataset = TEST(data_dir="../dataset/celeba-pbr/masked_rgb")
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=1,
                             pin_memory=True)
    
    answer_dict = dict()
    dataset_iter = iter(test_loader)
    progress_bar = tqdm(range(len(dataset_iter)), ncols=90)
    for step in progress_bar:
        
        # Load data
        data = next(dataset_iter)
        
        prompt_list = data["prompt"]
        prepare_list =[]
        for index, prompts in enumerate(zip(prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4])):
            
            pil_img = tvf.to_pil_image(data['rgb'][index].permute(2,0,1))
        
            # prompt
            if cot:    # 链式思考版本(示例)
                conversation =[
                    {"role": "<|User|>",        "content":"<image>\n<lref|>What's the person's gender?<l/ref|>"},
                    {"role": "<|Assistant|>",   "content": prompts[0]},
                    {"role": "<|User|>",        "content":"<image>\n<|ref|>And approximate age?<|/ref|>"},
                    {"role":"<|Assistant|>",    "content": prompts[2]},
                    {"role": "<|User|>",        "content":(
                        "<image>\n<|ref|>Summarise the facial appearance including gender, "
                        "age and key features in one sentence.<|/ref|>")},
                    {"role":"<|Assistant|>",    "content":""}
                ]
                images=[pil_img]*3
            else:      # 单轮版本(最常用)
                conversation =[
                    {"role": "<|User|>",
                    "content":(
                        "<image>\n<|ref|>Using one sentence, summarise the person's"
                        "facial features: gender,possible age, and appearance.<|/ref|>"
                    )},
                    {"role": "<|Assistant|>",   "content":""}
                ]
                images=[pil_img]
        
            prepare_list.append(vl_chat_processor.process_one(
                prompt=None,
                conversations=conversation,
                images=images,
                apply_sft_format=False,
                inference_mode=True,
                system_prompt=""
            ))
        
        prepare_inputs = vl_chat_processor.batchify(prepare_list).to(vl_gpt.device)
        prepare_inputs["images_seq_mask"] = prepare_inputs["images_seq_mask"].bool()
        
        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # run the model to get the response
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        answer_dict[data['file_index'][0]] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Save json data
    with open('answer.json', 'w', encoding='utf-8') as outfile:
        json_obj = json.dumps(answer_dict, indent=4, ensure_ascii=False)
        outfile.write(json_obj)

if __name__ == '__main__':
    
    with torch.no_grad():
        test(cot=True)
