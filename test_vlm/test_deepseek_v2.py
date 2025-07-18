import os, json, torch, warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images

# ② 你的自定义数据集
from test_loader import TEST

MODEL_DIR = "/data2/liuqianying/shading-diffusion-master/vlm-hf/deepseek-ai/deepseek-vl2"   # tiny / small / base / large
DATA_DIR = "../dataset/celeba-pbr/masked_rgb"
DEVICE   = "cuda:4"

def test(cot: bool = False, save_path: str = "answer_cot.json"):
    # 一行搞定 processor & model
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(MODEL_DIR)
    tokenizer = processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to(DEVICE).eval()

    # DataLoader
    dataloader = DataLoader(TEST(data_dir=DATA_DIR), batch_size=1, shuffle=False, num_workers=1)
    
    answer_dict = {}
    for data in tqdm(dataloader, ncols=100):
        prompt_list = data["prompt"]
        prepare_list = []
        for index, prompts in enumerate(zip(prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4])):

            # 把 tensor → PIL
            pil_img = tvf.to_pil_image(data["rgb"][index].permute(2, 0, 1))

            # print(prompts)

            # --------  prompt -----------
            if cot:          # 链式思考版本（示例）
                conversation = [
                        # ========= system =========
                    {
                    "role": "<|User|>",
                    "content": (
                        "You are a vision-language assistant. "
                        "When asked to summarise a face, output **one paragraph ≤ 200 words**. "
                        "The paragraph **must start** with a clause formatted like:\n"
                        "\"The individual in the image is a <**young / middle-aged / older** (pick one)> <male / female** (pick one)> in the age range of <low>-<high>."
                        # "- Replace <age_group> with **young / middle-aged / older** (pick one).\n"
                        # "- Replace <gender> with **male / female** (pick one).\n"
                        # "- Replace <low>-<high> with the numeric range (e.g. 20-30).\n"
                        # "Never output angle brackets or the words <age_group>, <gender>, <low>, <high>.\n"
                        "After that opening sentence, you have given into a single, coherent paragraph that fully describes the "
                        "person’s appearance. Make it suitable as a generation prompt for a diffusion model which can generate the image g-buffer figure, mentioning geometry "
                        "features, and texture details. "
                        "do not repeat any phrase."
                    )
                    },

                    # ========= user =========
                    # ① 性别与年龄
                    {"role": "<|User|>",
                    "content": (
                        "<image>\n"
                        "<|ref|>Please identify the person’s gender and approximate age.<|/ref|>"
                    )},
                    {"role": "<|Assistant|>", "content": ""},

                    # ② 脸型与整体轮廓
                    {"role": "<|User|>",
                    "content": "Describe the overall face shape and facial contour (jawline, cheekbones, forehead)."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ③ 五官比例与对称性
                    {"role": "<|User|>",
                    "content": "How are the facial features proportioned? Comment on spacing between eyes, nose, mouth and overall symmetry."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ④ 眼睛与眉毛
                    {"role": "<|User|>",
                    "content": "Describe the eyes and eyebrows: eye shape/size, eyelid type, eye-corner tilt, iris colour; eyebrow shape and density."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑤ 鼻子形态
                    {"role": "<|User|>",
                    "content": "Describe the nose: bridge height, length, width of the nostrils, and tip shape."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑥ 嘴巴与下巴
                    {"role": "<|User|>",
                    "content": "Describe the mouth and chin: lip thickness/shape, mouth-corner tilt, chin type and jaw definition."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑦ 肤色与整体质感（宏观）
                    # {
                    #     "role": "<|User|>",
                    #     "content": (
                    #         "Describe the skin colour and overall surface quality:\n"
                    #         "• Skin tone — name the overall shade (e.g., pale ivory, light olive, medium tan, deep brown) "
                    #         "and note the undertone (cool/pink, neutral, or warm/golden).\n"
                    #         "• Overall finish — say whether the skin looks matte, dewy, oily, or dry, without listing fine details."
                    #     )
                    # },
                    {
                        "role": "<|User|>",
                        "content": (
                            "Describe the skin colour and overall surface quality."
                        )
                    },
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑧ 皮肤细节与瑕疵
                    # {"role": "<|User|>",
                    # "content": "Describe all visible skin-texture details, separating them by type and location:\n"
                    #     "• Wrinkles — note depth and where they appear (e.g., horizontal lines on the forehead, crow’s-feet around the eyes).\n"
                    #     "• Pores / pits — mention if pores are enlarged or if there are acne-like depressions, and where.\n"
                    #     "• Discolorations — freckles, moles, scars, or redness; state approximate size and position."
                    #     },
                    # {"role": "<|Assistant|>", "content": ""},

                    {
                    "role": "<|User|>",
                    "content": (
                        "In 1–2 sentences, describe visible skin-texture details:\n"
                        "• Wrinkles — depth and location\n"
                        "• Pores / pits — enlarged or normal, location\n"
                        "• Discolorations — freckles, moles, scars, redness; size and location"
                    )
                    },
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑨ 头发与面部毛发
                    {"role": "<|User|>",
                    "content": "Describe hair and facial hair: hair colour, length, texture, hairline; presence and style of beard/moustache."},
                    {"role": "<|Assistant|>", "content": ""},

                    # ⑩ 整合描述 (生成最终 prompt)
                    {"role": "<|User|>",
                    "content": (
                        "Combine all the details you have given into a single, coherent paragraph (≤ 200 words) that fully describes the "
                        "person’s appearance. Make it suitable as a generation prompt for a diffusion model which can generate the image g-buffer figure, mentioning gender, age range, geometry "
                        "features, and texture details.\n"
                        "• Do **not** repeat any phrase.\n"
                        "• Each attribute only once.\n"
                        "• Keep the language natural."
                    )},
                    {"role": "<|Assistant|>", "content": ""}
                ]

                images = [pil_img]
            # elif cot2:          # 链式思考版本（示例）
            #     conversation = [
            #         {"role": "<|User|>",      "content": "<image>\n<|ref|>What's the person's gender?<|/ref|>"},
            #         {"role": "<|Assistant|>", "content": prompts[0]},
            #         {"role": "<|User|>",      "content": "<image>\n<|ref|>And approximate age?<|/ref|>"},
            #         {"role": "<|Assistant|>", "content": prompts[1]},
            #         {"role": "<|User|>",      "content": (
            #             "<image>\n<|ref|>Summarise the facial appearance including gender, "
            #             "age and key features in one sentence.<|/ref|>")},
            #         {"role": "<|Assistant|>", "content": ""}
            #     ]
            #     images = [pil_img]*3
            else:            # 单轮版本（最常用）
                conversation = [
                    {"role": "<|User|>",
                    "content": (
                        "<image>\n<|ref|>Using one sentence, summarise the person's "
                        "facial features: gender, possible age, and appearance.<|/ref|>"
                    )},
                    {"role": "<|Assistant|>", "content": ""}
                ]
                images = [pil_img]
            
            prepare = processor.process_one(
                prompt=None,
                conversations=conversation,
                images=images,
                apply_sft_format=False,
                inference_mode=True,
                system_prompt=""
            )
            prepare_list.append(prepare)
        
        prepares = processor.batchify(prepare_list)
        prepares = prepares.to(DEVICE)
        prepares["images_seq_mask"] = prepares["images_seq_mask"].bool()

        # multimodal pre-processing
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepares)

        # generation
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepares.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=False,
            use_cache=True,
            temperature=0.4,
            top_p=0.9)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        print(answer)

        answer_dict[data["file_index"][0]] = answer

    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with torch.inference_mode():
        test(cot=True,save_path= "/data2/liuqianying/shading-diffusion-master/test_vlm/answer_cot.json")