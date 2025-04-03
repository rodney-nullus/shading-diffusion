import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List, Tuple, Union
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as tvf

from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

class VLM(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        
        # Specify the path to the model
        model_path = "deepseek-ai/deepseek-vl2-tiny"
        
        # Load model
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).eval()
        
        # VLM settings
        self.max_new_tokens = configs.max_new_tokens
        self.do_sample = configs.do_sample
        self.use_cache = configs.use_cache
        
        self.system_prompt=""
        self.conversation_context = []
    
    def forward(self, processed_input, temperature=0.4, top_p=0.9):
        
        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**processed_input)
        
        # run model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=processed_input.attention_mask,
            pad_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            use_cache=self.use_cache,
            temperature=temperature,
            top_p=top_p
        )
        
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return answer
    
    def tokenize(self, conversations: List, images: List):
        
        processed_input = self.vl_chat_processor(
            conversations=conversations,
            images=images,
            force_batchify=True,
            system_prompt=self.system_prompt
        )
        
        return processed_input
    
    def tensors2pils(self, images: Tensor):
        
        pil_list = []

        for idx in range(images.shape[0]):
            pil_list.append(tvf.to_pil_image(images[idx]))

        return pil_list