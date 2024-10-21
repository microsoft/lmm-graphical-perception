"""This is used for the pipeline for chart reasoning task generation and grading (2d chart encoding)."""
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
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
# from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logger = logging.getLogger(__name__)

### local imports

APP_ROOT = Path(Path(__file__).parent.parent).absolute()
sys.path.append(os.path.abspath(APP_ROOT))

from .utils.clients import *
from .utils.data_utils import *
from .utils.vistext_gen_utils import *

from .module.task_generation import ChartReasoningTaskAgent
from .module.text_grading import TextGradingAgent
from .module.visual_reasoning import ChartVisualReasoningAgent
from .utils.chart_edit_utils import *


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

reading_guidance_dict = {
    'size': "The chart shows numbers as different circle sizes only, larger circle means larger number",
    'color': "The chart shows numbers as different color depth/saturation only, deeper color means larger number",
    'rule': "The chart shows numbers as different rule length, longer rule means larger number",
    'bar_color': "The chart shows numbers as the color of the bar and the bar length, deeper color means larger number and longer bar means larger number",
    'scatter_size': "The chart shows numbers as the size of the scatter points, larger size means larger number and higher position of scatters means larger number",
    'unaligned_rule': "The chart shows numbers as different rule length only, longer rule means larger number, the left most or the upmost rule is the length of average number"
}


class ChartReasoningPipeline(object):

    def __init__(self, 
                 vistext_data_dir,
                 pipeline_output_root,
                 seed=20240419,
                 dataset_size=5,  # each task has 5 data points
                 chart_type_list=['bar', 'bar_anno', 'line', 'line_anno', 'scatter', 'scatter_anno', 'table', 'pie',
                                  'size', 'color', 'rule', 'bar_color', 'scatter_size', 'unaligned_rule']):

        self.seed = seed
        self.chart_type_list = chart_type_list
        self.dataset_size = dataset_size

        self.vistext_data_dir = vistext_data_dir
        self.vistext_refined_data_dir = os.path.join(pipeline_output_root, "01-vistext-refined-data")
        self.generated_tasks_dir =  os.path.join(pipeline_output_root, "02-generated-tasks")
        self.chart_reasoning_output_dir = os.path.join(pipeline_output_root, "03-visual-chart-reasoning-output")
        self.graded_reasoning_output_dir = os.path.join(pipeline_output_root, "04-text-grading-output")
        self.graded_reasoning_output_md_dir = os.path.join(pipeline_output_root, "05-text-grading-output-md")

        # create the dir if not exists
        for dirname in ["", "01-vistext-refined-data", "02-generated-tasks", "03-visual-chart-reasoning-output", "04-text-grading-output", "05-text-grading-output-md"]:
            if not os.path.exists(os.path.join(pipeline_output_root, dirname)):
                os.makedirs(os.path.join(pipeline_output_root, dirname))

        logging.basicConfig(filename=os.path.join(pipeline_output_root, 'agents.log'), level=logging.INFO)


    def refine_vistext_data(self, debug=False):
        # refine vistext dataset to obtain

        path_scenegraph = os.path.join(self.vistext_data_dir, "scenegraphs")
        vl_spec_dir =  os.path.join(self.vistext_data_dir, "vl_spec")

        data_L1 = []
        n_success = 0
        n_error = 0
        n_skipped = 0

        files_sg = list(filter(lambda k: 'json' in k, os.listdir(path_scenegraph)))
        files_vl_spec = list(filter(lambda k: 'json' in k, os.listdir(vl_spec_dir)))
        files_sg.sort()
        files_vl_spec.sort()

        assert(len(files_sg) == 8822) ## unit test check length
        assert(len(files_sg) == len(files_vl_spec)) ## unit test check length

        # extract data as dict for manipulation
        def custom_parse_all_dt(obj):
            chart_title = parse_title(obj)[1]
            chart_x = parse_axes(obj)[5]
            chart_y = parse_axes(obj)[11]
            data_schema = (chart_x, chart_y)
            chart_data = parse_marks_dt(obj, chart_x, chart_y)
            chart_data_parsed = " ".join([x[0]+" "+x[1] for x in chart_data])
            parsed = chart_title + " <s> " + chart_x + " " + chart_y + " " + chart_data_parsed
            return parsed, data_schema, chart_data

        # process each file to extract data, vl_spec an others
        for file, vl_spec_file in tqdm(zip(files_sg, files_vl_spec), total=len(files_sg)):
            
            task_id = file.replace(".json","")

            filepath = os.path.join(path_scenegraph, file)
            vl_spec_file_path = os.path.join(vl_spec_dir, vl_spec_file)

            refined_file = os.path.join(self.vistext_refined_data_dir, f'{task_id}.json')

            if os.path.exists(refined_file):
                logger.info(f"[info] refined task {task_id} already exists, skip")
                n_skipped += 1
                continue
            
            with open(filepath) as f, open(vl_spec_file_path) as vl_f:
                read_sg = json.load(f)
                vl_spec = json.load(vl_f)
                try:
                    parsed_sg = parse_all_sg(read_sg)
                    parsed_dt, data_schema, chart_data = custom_parse_all_dt(read_sg)
                    parsed_metadata = parse_all_sg(read_sg,src=False)
                    parsed_caption, parsed_metadatadict = generate_caption(*parsed_metadata)

                    data_dict = []
                    for data_tuple in chart_data:
                        data_obj = {}
                        for i, key in enumerate(data_schema):
                            data_obj[key] = data_tuple[i]
                        data_dict.append(data_obj)

                    # data_dict
                    data_dict = data_dict[:self.dataset_size]  # only keep the first several data points to ensure the task is simple enough.
                    data_dict = basic_data_dict_cleaning(data_dict)
                    data_L1.append((task_id, 
                                    parsed_sg, parsed_dt, parsed_caption, 
                                    parsed_metadata, data_dict, vl_spec))
                    
                    vistext_task_dict = {
                        "img_id": task_id,
                        "scenegraph": parsed_sg,
                        "datatable": parsed_dt,
                        "caption_L1": parsed_caption,
                        "L1_properties": parsed_metadata,
                        "data_dict": data_dict,
                        "vl_spec": vl_spec
                    }

                    with open(refined_file, 'w') as out_f:
                        json.dump(vistext_task_dict, out_f, indent=4)
                    
                    if debug:
                        logger.debug(f"Parsed Scenegraph: {parsed_sg}\n")
                        logger.debug(f"Parsed Datatable: {parsed_dt}\n")
                        logger.debug(f"Parsed L1 Metadata: {parsed_metadata}")
                        #display(Image.open(os.path.join("../data/images", file.replace(".json",".png"))))
                        logger.debug(f"Parsed Caption: {parsed_caption}")
                        
                    n_success += 1
                except Exception as e:
                    n_error += 1 
                    print(e)
                    
        logger.info(f"Successes: {n_success}")
        logger.info(f"Errors: {n_error}")
        logger.info(f"Skipped: {n_skipped}")


    def task_generation(self, id_lists=None, sample_size=50):
        # generate tasks from refined data

        def task_generator(data_dict, vl_spec):
            clients = OpenAIClients()
            agent = ChartReasoningTaskAgent(clients.gpt4_client, model=clients.gpt4_model)
            tasks = []
            
            results = agent.run(data_dict, vl_spec, n=1)
            for r in results:
                if r['status'] == 'ok':
                    tasks.append(r['content']['tasks'])
            
            return tasks

        np.random.seed(self.seed)

        vistext_refined_data_files = sorted(os.listdir(self.vistext_refined_data_dir))
        vistext_refined_data_samples = np.random.choice(vistext_refined_data_files, sample_size)

        if id_lists is not None:  # todo, check
            vistext_refined_data_samples = [fname for fname in vistext_refined_data_files if fname in id_lists]

        for fname in tqdm(vistext_refined_data_samples):
            fpath = os.path.join(self.vistext_refined_data_dir, fname)

            with open(fpath) as f:
                task_source = json.load(f)

            task_id = task_source['img_id']
            logger.info(f"[task_generation] processing task {task_id}")
            out_task_json_fname = os.path.join(self.generated_tasks_dir, f'{task_id}.task.json')
            task_exists = False
            if os.path.exists(out_task_json_fname):
                task_exists = True
                all_image_exists = True
                for chart_type in self.chart_type_list:
                    img_path = os.path.join(self.generated_tasks_dir, f"{task_id}_{chart_type}.png")
                    if not os.path.exists(img_path):
                        all_image_exists = False
                        break
                if all_image_exists:
                    logger.info(f"{task_id} already exists.")
                    continue


            data_dict = task_source['data_dict']
            if not task_exists:
                tasks = task_generator(data_dict, task_source['vl_spec'])
                with open(out_task_json_fname, 'w') as g:  # save the task json
                    json.dump(tasks, g, indent=4)

            for chart_type in self.chart_type_list:
                # RQ1
                if chart_type == 'bar':
                    tmp_source = edit_bar_wo_annotation(copy.deepcopy(task_source))
                elif chart_type == 'bar_anno':
                    tmp_source = edit_bar_w_annotation(copy.deepcopy(task_source))
                elif chart_type == 'line':
                    tmp_source = edit_line_wo_annotation(copy.deepcopy(task_source))
                elif chart_type == 'line_anno':
                    tmp_source = edit_line_w_annotation(copy.deepcopy(task_source))
                elif chart_type == 'scatter':
                    tmp_source = edit_scatter_wo_annotation(copy.deepcopy(task_source))
                elif chart_type == 'scatter_anno':
                    tmp_source = edit_scatter_w_annotation(copy.deepcopy(task_source))
                elif chart_type == 'table':
                    edit_table(copy.deepcopy(task_source), os.path.join(self.generated_tasks_dir, f"{task_id}_{chart_type}.png"))
                elif chart_type == 'pie':
                    tmp_source = edit_pie_w_annotation(copy.deepcopy(task_source))

                # RQ2
                elif chart_type == 'size':
                    tmp_source = edit_size(copy.deepcopy(task_source))
                elif chart_type == 'color':
                    tmp_source = edit_color(copy.deepcopy(task_source))
                elif chart_type == 'rule':
                    tmp_source = edit_rule_color(copy.deepcopy(task_source))
                elif chart_type == 'bar_color':
                    tmp_source = edit_bar_color(copy.deepcopy(task_source))
                elif chart_type == 'scatter_size':
                    tmp_source = edit_scatter_size(copy.deepcopy(task_source))
                elif chart_type == 'unaligned_rule':
                    tmp_source = edit_unaligned_rule(copy.deepcopy(task_source))
                else:
                    raise ValueError(f"chart_type {chart_type} not supported")

                if chart_type != 'table':
                    save_vl_png(tmp_source['vl_spec'], tmp_source['data_dict'], save_file=os.path.join(self.generated_tasks_dir, f"{task_id}_{chart_type}.png"))
                    

    def visual_chart_reasoning(self, use_reading_guidance=True):
        
        def chart_QA(img_path, tasks, reading_guidance=None):
    
            clients = OpenAIClients()
            agent = ChartVisualReasoningAgent(clients.gpt4_client, model=clients.gpt4_vision_model)
            
            results = agent.run(img_path, tasks, n=1, reading_guidance=reading_guidance)

            return results[0]['content']

        task_dir = self.generated_tasks_dir
        task_ids = [fname.split(".")[0] for fname in os.listdir(task_dir) if fname.endswith('.task.json')]
        logger.info(f"[visual_reasoning] {len(task_ids)} tasks to process.")

        for task_id in tqdm(task_ids):
            valid_example = True
            for chart_type in self.chart_type_list:
                img_path = os.path.join(task_dir, f"{task_id}_{chart_type}.png")
                if not os.path.exists(img_path):
                    valid_example = False
                    break
            if not valid_example:
                for chart_type in self.chart_type_list:
                    img_path = os.path.join(task_dir, f"{task_id}_{chart_type}.png")
                    if os.path.exists(img_path):
                        os.remove(img_path)
                img_path = os.path.join(self.generated_tasks_dir, f"{task_id}.png")
                if os.path.exists(img_path):
                    os.remove(img_path)
                task_file = os.path.join(task_dir, f"{task_id}.task.json")
                if os.path.exists(task_file):
                    os.remove(task_file)
                    logger.info(f"[visual_reasoning] {task_id} removed.")
                    print(f"Note: [visual_reasoning] {task_id} removed.")
                continue

            for chart_type in self.chart_type_list:
                img_path = os.path.join(task_dir, f"{task_id}_{chart_type}.png")
                task_file = os.path.join(task_dir, f"{task_id}.task.json")
                out_log_file = os.path.join(self.chart_reasoning_output_dir, f"{task_id}.{chart_type}.ans.json")

                logger.info(f'[chart_reasoning] processing {task_id}')
                
                if os.path.exists(out_log_file):
                    logger.info(f'{task_id}.{chart_type} exists, skip')
                    continue
                logger.info(f"[visual_reasoning] {task_id} {chart_type} processing..")
                try:
                    with open(task_file, 'r') as f:
                        tasks = json.load(f)[0]
                        task_input = [task['description'] for task in tasks]

                    if use_reading_guidance:
                        reading_guidance = reading_guidance_dict[chart_type]
                    else:
                        reading_guidance = None

                    answers = chart_QA(img_path, task_input, reading_guidance=reading_guidance)
                    
                    if len(answers) == 0:
                        continue

                    for task, answer in zip(tasks, answers):
                        task['answer'] = answer['answer']
                        task['reasoning'] = answer['reasoning']
                
                    with open(out_log_file, 'w') as g:
                        json.dump(tasks, g, indent=4)
                except Exception as e: 
                    logger.info(f"error: {task_id} failed")
                    logger.info(repr(e))
                    pass


    def grade_with_text_agent(self):
        
        def chart_reasoning_eval_call(data_title, data_dict, formatted_answers):

            clients = OpenAIClients()
            agent = TextGradingAgent(clients.gpt4_client, model=clients.gpt4_vision_model)            

            results = agent.run(data_title, data_dict, formatted_answers, n=1)
            return results[0]['content']

        
        answer_dir = self.chart_reasoning_output_dir
        graded_output_dir = self.graded_reasoning_output_dir

        task_ids = [fname.split(".")[0] for fname in os.listdir(answer_dir) if fname.endswith('.ans.json')]

        for task_id in tqdm(task_ids):

            logger.info(f'[grade_reasoning_results] processing task {task_id}..')

            # ensure every file exists
            all_charts_exist_flag = True
            for chart_type in self.chart_type_list:
                ans_file = os.path.join(answer_dir, f"{task_id}.{chart_type}.ans.json")
                if not os.path.exists(ans_file):
                    all_charts_exist_flag = False
                    break
                if not os.path.exists(os.path.join(self.generated_tasks_dir, f"{task_id}_{chart_type}.png")):
                    all_charts_exist_flag = False
                    break
            
            if not all_charts_exist_flag:
                logger.info(f"[info] {task_id} not all charts exist, skip")
                print(f"[info] {task_id} not all charts exist, skip")
                continue
            if not os.path.exists(os.path.join(self.generated_tasks_dir, f"{task_id}.task.json")):
                logger.info(f"[info] {task_id} task.json not exist, skip")
                print(f"[info] {task_id} task.json not exist, skip")
                continue

            task_metadata_file = os.path.join(self.vistext_refined_data_dir, f"{task_id}.json")
            for chart_type in self.chart_type_list:
                ans_file = os.path.join(answer_dir, f"{task_id}.{chart_type}.ans.json")
                
                eval_output_file = os.path.join(graded_output_dir, f"{task_id}.{chart_type}.grade.json")
                
                if os.path.exists(eval_output_file):
                    logger.info(f'{task_id}.{chart_type} already exists, skip')
                    continue
                if not os.path.exists(os.path.join(self.generated_tasks_dir, f"{task_id}_{chart_type}.png")):
                    continue
                try:
                    with open(task_metadata_file, 'r') as f:
                        task_metadata = json.load(f)
                    with open(ans_file, 'r') as f:
                        answer_list = json.load(f)
                except:
                    continue

                #data_sample = task_metadata['data_dict'][:20] if len(task_metadata['data_dict']) > 20 else task_metadata['data_dict']
                #table_sample_strings = f'sample:\n\n```\n{pd.DataFrame(data_sample).to_csv(sep="|")}......\n```'

                #display(IPython.display.Image(filename=img_path))
                #print(table_sample_strings)

                try:
                    formatted_answer = [
                        {
                            "task": answer['description'],
                            "task_type": answer['type'],
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
                    pass


    def graded_output_to_md(self, output_md_dir=None, graded_output_dir=None):

        if output_md_dir is None:
            output_md_dir = self.graded_reasoning_output_md_dir
        else: # new dir may not exist
            if not os.path.exists(output_md_dir):
                os.makedirs(output_md_dir, exist_ok=True)
        if graded_output_dir is None:
            graded_output_dir = self.graded_reasoning_output_dir
        

        graded_task_ids = [fname.split(".")[0] for fname in os.listdir(self.graded_reasoning_output_dir) if fname.endswith('.grade.json')]

        for task_id in tqdm(graded_task_ids):
            if not os.path.exists(os.path.join(self.generated_tasks_dir, f"{task_id}.task.json")):
                logger.info(f"[info] {task_id} task.json not exist, skip")
                print(f"[info] {task_id} task.json not exist, skip")
                continue

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
                    os.makedirs(output_img_dir)
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