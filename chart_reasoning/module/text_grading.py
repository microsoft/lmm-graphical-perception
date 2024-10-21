import json
import openai
import os

import tempfile
import copy
import vl_convert as vlc
import numpy as np

import pandas as pd

from ..utils.clients import *
from ..utils.data_utils import *

import pathlib
import logging
logger = logging.getLogger(__name__)

example_image_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'example_images')

system_message = {
    'role': 'system',
    'content': [ {
        'type': 'text',
        'text': '''You are a teacher to grade students' answers.
We will provide you a dataset, a list of tasks and student answers. Your goal is to use the dataset to evaluate if the student's answer is correct.
In order to form a good judgement, you should first use the dataset to derive your answer, and then compare it with the students asnwer.

When you generate the referenece answer:
* If the task asks for a value, provide the value directly.
* If the task asks for trend, answer it with one of "increasing", "decreasing" if the general trend point to the direcrtion, otherwise provide "unclear".
* Provide a brief reasoning of how you come up with your answer in "reasoning" part.
* If you cannot answer a question, provide "I don't know" as the answer, try not to provide a wrong answer.

* The student_answer_correctness should include the grading results of the student answer, it should be one of the following options: 
        - correct (note that if the student answer (value) is an approximation of your reference answer and it's within 5% of the reference answer, then it's also correct)
        - incorrect
        - fair (note that if the student answer (value) is an approximation of your reference answer and it's within 20% of the reference answer, then it's also fair)
        - skipped (if the student skipped the answer)
        - n/a (if the task does not make sense or not answerable with the given dataset)

Grade student questions based on [Data], [Tasks & Student Answers].
The output json should have the format [{"reasoning": ..., "reference answer": ..., "comparison_with_student_answer": ..., "student_answer_correctness": ...}, ....].
'''}]}


def vegalite_to_png_url(input_table, vl_spec):
    # create a temp file to save vega lite plot

    vl_spec_copy = copy.deepcopy(vl_spec)
    vl_spec_copy['data'] = {"values": input_table['rows']}
    
    png_data = vlc.vegalite_to_png(vl_spec=vl_spec_copy, scale=1)

    tmp = tempfile.NamedTemporaryFile(suffix='.png')
    with open(tmp.name, 'wb') as f:
        f.write(png_data)
        #display(Image(filename=tmp.name))
        png_url_data = convert_image_to_url(tmp.name)
    
        return png_url_data

class TextGradingAgent(object):

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def run(self, data_metadata, data_dict, tasks, n=1):
        
        def derive_quick_field_summary(data_dict):
            df = pd.DataFrame(data_dict)
            cols = df.columns
            field_summary = ""
            for c in cols:
                try:
                    new_col = pd.to_numeric(df[c])
                    precision = np.max([len(v.split('.')[-1]) if '.' in v else 0 for v in df[c]])
                    df[c] = new_col
                except:
                    precision = 0
                    pass
                info = df[c].describe()
                info_str = f"    {info.name} --- "
                for i, v in zip(info.index, info.values):
                    if type(v) == np.float64:
                        v = np.round(v, precision)
                    info_str += f"{i}: {v}, "

                field_summary += f"{info_str}\n"

            field_summary = field_summary
            return field_summary

        
        field_summary = derive_quick_field_summary(data_dict)
        data_sample = data_dict[:100] if len(data_dict) > 100 else data_dict
        sample_prefix = "data sample (first 100 rows):\n\n" if len(data_dict) > 100 else "full data:\n\n"
        table_sample_strings = f'{sample_prefix}```{pd.DataFrame(data_sample).to_csv(sep="|")}\n```'

        
        query = {
            'role': 'user',
            'content': [ {
                    'type': 'text',
                    'text': f'''[Title]\n\n{data_metadata}\n\n[Field Summary]\n{field_summary}\n\n[Data]\n\n{table_sample_strings}\n\n[Tasks & Student Answers]\n\n{json.dumps(tasks)}\n\n[Output]\n'''
                }
            ]
        }

        logger.info(">>> Chart Reasoning Eval Agent <<<\n")

        logger.info("##### [user query] #####")
        logger.info(query['content'][0]['text'])

        messages = [system_message, query]

        response = self.client.chat.completions.create(
                    model=self.model, messages = messages, temperature=0.7, max_tokens=4096,
                    top_p=0.95, n=n)

        logger.info(f"##### [agent responses] #####")
        
        candidates = []
        for choice in response.choices:
            
            logger.info(choice.message.content + "\n")
            
            blocks = extract_json_objects(choice.message.content + "\n")
            
            if len(blocks) > 0:
                result = {'status': 'ok', 'content': blocks[-1]}
            else:
                result = {'status': 'error', 'content': 'unable to extract json response from response'}
            
            # individual dialog for the agent
            result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]

            candidates.append(result)

        return candidates