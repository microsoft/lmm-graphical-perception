import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5,6'
"""
CUDA_VISIBLE_DEVICES=4 nohup python main_phi3_prob.py > log_phi3_prob.txt 2>&1 &
"""

from copy import deepcopy
import torch

import torch
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageStat
import json

PROMPT_TEMPLATE = 'Please give answer (number) directly and only: {}'

model_id = "microsoft/Phi-3.5-vision-instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda", torch_dtype="auto", cache_dir='/research/nfs_su_809/workspace/zhang.13253/hf_cache') # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)


input_root_dir = './data/annotated_importance_heatmap'

output_root_dir = './output/phi3_prob_output'

id_list = os.listdir(input_root_dir)


for id in tqdm(id_list):
    print("Running on id:", id)
    input_dir = os.path.join(input_root_dir, id)
    output_dir = os.path.join(output_root_dir, id)
    
    if id == '.DS_Store':
        continue

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 6:
        print("Output directory already exists for id:", id)
        continue
    os.makedirs(output_dir, exist_ok=True)
    try:
        task_list = json.load(open(os.path.join(input_dir, f'{id}.task.json')))
        task_list = task_list[0]
    except:
        print("Error: task.json not found for id:", id)
        continue
    # find the first Retrieve-Value task.
    for task in task_list:
        if task['type'] == 'Retrieve Value':
            question = task['description']
            break
    for chart_type in ['bar', 'bar_anno']:
        image_name = f'{id}_{chart_type}.png'
        image = Image.open(os.path.join(input_dir, image_name))
        image = image.resize((336, 336))
        input_text = PROMPT_TEMPLATE.format(question)
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{input_text}"}, 
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 
        generation_args = {
            "max_new_tokens": 10, 
            "temperature": 0.0, 
            "do_sample": False, 
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_attentions": True,
            
        } 
        with torch.no_grad():
            returned = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

        generate_ids = returned['sequences'].detach().cpu().numpy()
        # remove input tokens 
        # import pdb; pdb.set_trace()
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        original_response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0] 

        scores = returned['scores']
        desired_ans_scores = []
        desired_ans_indexes = []
        for score in returned['scores'][:-2]: # each step excepts the eos tokens
            desired_ans_scores.append(torch.max(score).detach().cpu().numpy().tolist())
            desired_ans_indexes.append(torch.argmax(score).detach().cpu().numpy().tolist())

        del returned
        del inputs

        crop_size = (28, 28)
        image_size = (336, 336)
        crop_positions = [(x, y) for x in range(0, image_size[0], crop_size[0]) for y in range(0, image_size[0], crop_size[0])]

        # Store scores for each masked crop
        crop_scores = []


        def mask_crop(image, crop_position, crop_size=(14, 14)):
            # get the most common color in the entire image (background)
            # Get the most common pixel color (RGB) in the image
            pixels = list(image.getdata())
            background_color = max(set(pixels), key=pixels.count)
            # print("background_color:", background_color)

            masked_image = image.copy()
            draw = ImageDraw.Draw(masked_image)
            x, y = crop_position
            current_target_mask_region = (x, y, x + crop_size[0], y + crop_size[1])
            # if the pixels in the original region are all the same (so they are background already), then skip
            if len(set(masked_image.crop(current_target_mask_region).getdata())) == 1:
                return False, masked_image
            else:
                draw.rectangle([x, y, x + crop_size[0], y + crop_size[1]], fill=background_color)
                return True, masked_image


        for crop_position in tqdm(crop_positions):
            masked, masked_image = mask_crop(image, crop_position, crop_size)
            if not masked:
                crop_scores.append({
                    "crop_position": crop_position,
                    "scores": desired_ans_scores,
                    "response": original_response,
                    "original": True
                })
                continue


            inputs = processor(prompt, [masked_image], return_tensors="pt").to("cuda:0")

            with torch.no_grad():
                returned = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
            generate_ids = returned['sequences']
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            # print(response)
            
            scores = returned['scores']
            crop_ans_scores = []
            # if scores are not the same length with desired_ans_indexes, then shink to the same length
            min_length = min(len(desired_ans_indexes), len(scores))

            for i in range(min_length):
                crop_ans_scores.append(scores[i][0, desired_ans_indexes[i]].detach().cpu().numpy().tolist())

            crop_scores.append({
                "crop_position": crop_position,
                "scores": crop_ans_scores,
                "response": response,
                "original": False
            })        

        attention_method = "first_token"
        # attention_method = "mean"
        only_top_n = None
        min_len = None
        # min_len = 10


        # Now, use the crop_scores to draw attention maps

        attention_map = np.zeros((image_size[0]//crop_size[0], image_size[0]//crop_size[0]))

        top_scores = []
        top_locations = []
        for crop_score in crop_scores:
            x, y = crop_score['crop_position']
            if attention_method == "first_token":
                attention_map[y // crop_size[0], x // crop_size[0]] = abs(crop_score['scores'][0] - desired_ans_scores[0])
            elif attention_method == "mean":
                if min_len is not None:
                    min_length = min(len(crop_score['scores']), len(desired_ans_scores), min_len)
                else:
                    min_length = min(len(crop_score['scores']), len(desired_ans_scores))
                for i in range(min_length):
                    attention_map[y // crop_size[0], x // crop_size[0]] += abs(crop_score['scores'][i] - desired_ans_scores[i])
                attention_map[y // crop_size[0], x // crop_size[0]] /= min_length

        # only keep top n and remove other scores
        if only_top_n is not None:  
            top_n_indexes = np.argsort(attention_map.flatten())[-only_top_n:]
            for i in range(attention_map.size):
                if i not in top_n_indexes:
                    attention_map[i // attention_map.shape[0], i % attention_map.shape[0]] = 0


        attention_map = attention_map / np.max(attention_map)

        # Resize the attention map to match the original image size
        attention_map_resized = np.kron(attention_map, np.ones((crop_size[0], crop_size[0])))

        print("Question:", question)
        print("Answer:", original_response)
        # save attention map
        plt.imshow(image)
        plt.imshow(attention_map_resized, cmap='hot', alpha=0.5)  # alpha controls the transparency of the heatmap
        plt.colorbar()  # Add a colorbar to show the intensity scale
        plt.axis('off')  # Turn off the axis
        plt.tight_layout()  # Adjust the layout
        plt.savefig(os.path.join(output_dir, f'{chart_type}_attention_map.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        # save all the scores and attention map
        with open(os.path.join(output_dir, f'{chart_type}_crop_scores.json'), 'w') as f:
            json.dump(crop_scores, f, indent=4)

        # save attention_map .json
        with open(os.path.join(output_dir, f'{chart_type}_attention_map.json'), 'w') as f:
            json.dump(attention_map.tolist(), f, indent=4)
        print("Done with id:", id, "chart_type:", chart_type)
    print("Finished running on id:", id)
