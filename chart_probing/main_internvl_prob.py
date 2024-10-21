import os
import sys
"""
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main_internvl_awq_prob.py > log_internvl_awq_prob.txt 2>&1 &
"""

from copy import deepcopy
import torch
torch.set_default_dtype(torch.bfloat16)
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageStat
import json

model = 'InternVL2-Llama3-76B-AWQ'
backend_config = TurbomindEngineConfig(model_format='awq', tp=4)
pipe = pipeline(model, backend_config=backend_config, log_level='ERROR')
PROMPT_TEMPLATE = 'Please give answer (number) directly and only: {}'

generation_config = GenerationConfig(
    max_new_tokens=10,
    temperature=0.0,
    top_p=0.0,
    top_k=40
)

input_root_dir = './data/annotated_importance_heatmap'

output_root_dir = './output/internvl_awq_prob_output'

id_list = os.listdir(input_root_dir)

# if len(sys.argv) > 1:
#     id_list = sys.argv[1:]

# if sys.argv[1] == 'reverse':
#     id_list = id_list[::-1]

for id in tqdm(id_list):
    print("Running on id:", id)
    input_dir = os.path.join(input_root_dir, id)
    output_dir = os.path.join(output_root_dir, id)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 6:
        print("Output directory already exists for id:", id)
        continue
    
    # if it's DS_Store, skip
    if id == '.DS_Store':
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
        # question = 'What was the total number of prescriptions in 2005?'
        input_text = PROMPT_TEMPLATE.format(question)


        # get original logits
        inputs = pipe.prepare_inputs((input_text, image))
        input_ids = inputs['input_ids']
        embeddings = inputs['input_embeddings']
        embedding_ranges = inputs['input_embedding_ranges']


        # get output ids
        out = pipe((input_text, image))
        original_output_ids = out.token_ids

        # get logits
        tmp_ids = deepcopy(input_ids)
        all_original_logits = []
        # Iterate over each generated token and accumulate logits
        for idx, token_id in enumerate(original_output_ids):
            tmp_ids[0].append(token_id)  # Append current token ID to the sequence
            
            # Get logits for the current sequence including the new token
            logits = pipe.get_logits(tmp_ids, embeddings, embedding_ranges)
            
            # Determine the correct position of the token in the concatenated logits
            token_position = len(tmp_ids[0]) - 1  # Position of the current token in the sequence
            
            # Extract the logits corresponding to the current token ID
            # Access the logits at the batch index 0, correct token position, and the specific token index
            all_original_logits.append(logits[0, token_position, token_id].detach().cpu().numpy().tolist())


        

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
                    "scores": all_original_logits,
                    "response": pipe.tokenizer.decode(original_output_ids),
                    "original": True
                })
                continue

            inputs = pipe.prepare_inputs((input_text, masked_image))
            input_ids = inputs['input_ids']
            embeddings = inputs['input_embeddings']
            embedding_ranges = inputs['input_embedding_ranges']

            # get logits
            out = pipe((input_text, masked_image))
            tmp_ids = deepcopy(input_ids)
            
            all_logits_on_masked_image = []

            for idx, token_id in enumerate(original_output_ids):
                tmp_ids[0].append(token_id)  # Append current token ID to the sequence

                # Get logits for the current sequence including the new token
                logits = pipe.get_logits(tmp_ids, embeddings, embedding_ranges)

                # Determine the correct position of the token in the concatenated logits
                token_position = len(tmp_ids[0]) - 1  # Position of the current token in the sequence

                # Extract the logits corresponding to the current token ID
                all_logits_on_masked_image.append(logits[0, token_position, token_id].detach().cpu().numpy().tolist())
            
            crop_scores.append({
                "crop_position": crop_position,
                "scores": all_logits_on_masked_image,
                "response": pipe.tokenizer.decode(original_output_ids),
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

        # import pdb; pdb.set_trace()
        for crop_score in crop_scores:
            x, y = crop_score['crop_position']
            if attention_method == "first_token":
                attention_map[y // crop_size[0], x // crop_size[0]] = abs(crop_score['scores'][0] - all_original_logits[0])
            elif attention_method == "mean":
                if min_len is not None:
                    min_length = min(len(crop_score['scores']), len(all_original_logits), min_len)
                else:
                    min_length = min(len(crop_score['scores']), len(all_original_logits))
                for i in range(min_length):
                    attention_map[y // crop_size[0], x // crop_size[0]] += abs(crop_score['scores'][i] - all_original_logits[i])
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
        print("Answer:", pipe.tokenizer.decode(original_output_ids))
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
