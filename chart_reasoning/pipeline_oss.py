"""This is used for the pipeline for graphical perception grading."""
import os
import re
import sys
import json
import copy
import tempfile
import base64

import pandas as pd
import numpy as np
import vl_convert as vlc
import dataframe_image as dfi
# from vega import VegaLite
# from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

### local imports

APP_ROOT = Path(Path(__file__).parent.parent).absolute()
sys.path.append(os.path.abspath(APP_ROOT))

from .utils.clients import *
from .utils.data_utils import save_vl_png, convert_image_to_url
from .module.text_grading import TextGradingAgent

# this is a specific cleaning for the dataset...
def basic_data_dict_cleaning(data_dict):
    if "Year" in data_dict[0].keys():
        year_values = [r["Year"] for r in data_dict]
        if all([("Dec 31, " in v) for v in year_values]):
            out_data_dict = []
            for r in data_dict:
                try:
                    new_val = r["Year"].replace("Dec 31, ", "")
                    r["Year"] = f"{int(new_val) + 1}"
                except:
                    pass
                out_data_dict.append(r)
            return out_data_dict
    return data_dict


class ChartReasoningOSSPipeline(object):

    def __init__(self, 
                 vistext_data_dir,
                 pipeline_output_root,
                 seed=20240419,
                 model_name=None,
                 chart_type_list=['bar', 'bar_anno', 'line', 'line_anno', 'scatter', 'scatter_anno', 'table', 'pie',
                                  'size', 'color', 'rule', 'bar_color', 'scatter_size', 'unaligned_rule']):

        self.seed = seed
        self.model_name = model_name
        if model_name is None:
            raise ValueError("model_name is required")
        
        self.chart_type_list = chart_type_list

        self.vistext_data_dir = vistext_data_dir
        self.vistext_refined_data_dir = os.path.join(pipeline_output_root, "01-vistext-refined-data")
        self.generated_tasks_dir =  os.path.join(pipeline_output_root, "02-generated-tasks")
        self.chart_reasoning_output_dir = os.path.join(pipeline_output_root, "03-oss-chart-reasoning-output", self.model_name)
        self.graded_reasoning_output_dir = os.path.join(pipeline_output_root, "04-oss-text-grading-output", self.model_name)
        self.graded_reasoning_output_md_dir = os.path.join(pipeline_output_root, "05-oss-text-grading-output-md", self.model_name)

        # create the dir if not exists
        for dirname in ["", "01-vistext-refined-data", "02-generated-tasks", "03-oss-chart-reasoning-output",
                        "04-oss-text-grading-output", "05-oss-text-grading-output-md"]:

            if not os.path.exists(os.path.join(pipeline_output_root, dirname)):
                os.mkdir(os.path.join(pipeline_output_root, dirname))
        
        for dirname in ["03-oss-chart-reasoning-output", "04-oss-text-grading-output", "05-oss-text-grading-output-md"]:
            if not os.path.exists(os.path.join(pipeline_output_root, dirname, self.model_name)):
                os.mkdir(os.path.join(pipeline_output_root, dirname, self.model_name))

        logging.basicConfig(filename=os.path.join(pipeline_output_root, 'agents_oss.log'), level=logging.INFO)

    def grade_with_text_agent(self):

        def chart_reasoning_eval_call(data_title, data_dict, formatted_answers):

            clients = OpenAIClients()
            agent = TextGradingAgent(clients.gpt4_client, model=clients.gpt4_vision_model)            
            
            processed_data_dict = basic_data_dict_cleaning(data_dict)

            results = agent.run(data_title, processed_data_dict, formatted_answers, n=1)
            return results[0]['content']

        answer_dir = self.chart_reasoning_output_dir
        graded_output_dir = self.graded_reasoning_output_dir
        
        task_ids = [fname.split(".")[0] for fname in os.listdir(answer_dir) if fname.endswith('.ans.json')]

        def process_task(task_id, vistext_refined_data_dir, answer_dir, graded_output_dir):
            logger.info(f'[grade_reasoning_results] processing task {task_id}..')

            task_metadata_file = os.path.join(vistext_refined_data_dir, f"{task_id}.json")
            for chart_type in self.chart_type_list:
                ans_file = os.path.join(answer_dir, f"{task_id}.{chart_type}.ans.json")
                eval_output_file = os.path.join(graded_output_dir, f"{task_id}.{chart_type}.grade.json")

                if os.path.exists(eval_output_file):
                    logger.info(f'{task_id} already exists, skip')
                    continue

                try:
                    task_metadata = json.load(open(task_metadata_file, 'r'))
                    answer_list = json.load(open(ans_file, 'r'))
                    true_answer_list = [answer for answer in answer_list if answer['task_id'].endswith(chart_type)]
                    answer_list = true_answer_list
                except Exception as e:
                    logger.info(f"error: loading {task_id} failed")
                    logger.info(repr(e))
                    continue

                try:
                    formatted_answer = [
                        {
                            "task": answer['description'],
                            "task_type": answer['question_type'],
                            "student_answer": answer['answer']
                        } for answer in answer_list
                    ]
                    processed_results = []
                    results = chart_reasoning_eval_call(" ".join(task_metadata['vl_spec']['title']), task_metadata['data_dict'], formatted_answer)
                    for i, result in enumerate(results):
                        clean_result = {
                            "task_id": task_id,
                            "task_type": formatted_answer[i]['task_type'],
                            "task": formatted_answer[i]['task'],
                            "student_answer": formatted_answer[i]['student_answer'],
                            "reference_answer": result['reference answer'],
                            "comparison_with_student_answer": result["comparison_with_student_answer"],
                            "student_answer_correctness": result["student_answer_correctness"],
                            "student_reasoning": answer_list[i]['reasoning'],
                            "reference_reasoning": result['reasoning']
                        }
                        processed_results.append(clean_result)

                    with open(eval_output_file, 'w') as g:
                        json.dump(processed_results, g, indent=4)
                except Exception as e:
                    logger.info(f"error: {task_id} failed")
                    logger.info(repr(e))
                    continue
    
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(process_task, task_id, self.vistext_refined_data_dir, answer_dir, graded_output_dir): task_id for task_id in task_ids}
            for future in tqdm(as_completed(futures), total=len(futures)):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.info(f"Task {task_id} generated an exception: {e}")


    def graded_output_to_md(self, output_md_dir=None, graded_output_dir=None):

        if output_md_dir is None:
            output_md_dir = self.graded_reasoning_output_md_dir
        else: # new dir may not exist
            if not os.path.exists(output_md_dir):
                os.mkdir(output_md_dir)
        if graded_output_dir is None:
            graded_output_dir = self.graded_reasoning_output_dir
        

        graded_task_ids = [fname.split(".")[0] for fname in os.listdir(self.graded_reasoning_output_dir) if fname.endswith('.grade.json')]

        for task_id in tqdm(graded_task_ids):

            ## raw data url
            for chart_type in self.chart_type_list:
                png = os.path.join(self.generated_tasks_dir, f'{task_id}_{chart_type}.png')
                task_chart_url = convert_image_to_url(png, 512)
                
                graded_task_file = os.path.join(self.graded_reasoning_output_dir, f'{task_id}.{chart_type}.grade.json')
                refined_task_file = os.path.join(self.vistext_refined_data_dir, f"{task_id}.json")
                # if graded_task_file not exist
                if not os.path.exists(graded_task_file):
                    continue

                output_md_file = os.path.join(output_md_dir, f'{task_id}.{chart_type}.grade.md')

                output_img_dir = os.path.join(output_md_dir, "images")
                if not os.path.exists(output_img_dir): 
                    os.mkdir(output_img_dir)
                # copy the image to the output dir

                output_png_file = os.path.join(output_img_dir, f'{task_id}_{chart_type}.png')
                os.system(f"cp {png} {output_png_file}")
                
                with open(refined_task_file) as f:
                    data_df = pd.DataFrame(json.load(f)['data_dict'])

                data_chart_str = f"""
<table>
<tr><th>Data</th><th>Chart</th></tr>
<tr><td>

{data_df.to_markdown()}

</td><td>

![Hello World](images/{task_id}_{chart_type}.png)

</td></tr> </table>    
        """
                
                md_str = f"### Task {task_id}\n\n{data_chart_str}\n\n"
                with open(graded_task_file) as f:
                    graded_tasks = json.load(f)
                    df = pd.DataFrame(graded_tasks)
                    df = df.drop(columns=['task_id'])
                    df = df.rename(columns={
                        "task": "task_________________________",
                        "comparison_with_student_answer": "comparison_with_student_answer",
                        "student_reasoning": "student_reasoning__________________________________________________",
                        "reference_reasoning": "reference_reasoning_________________________",
                    })
                    #display(df)
                    md_str += df.to_markdown()

                    with open(output_png_file, 'wb') as g:
                        #print(task_chart_url)
                        binary_data = base64.b64decode(task_chart_url[len("data:image/png;base64,"):])
                        g.write(binary_data)

                    with open(output_md_file, "w") as g:
                        g.write(md_str)