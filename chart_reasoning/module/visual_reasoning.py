import json
import openai
import os

import tempfile
import copy
import vl_convert as vlc

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
        'text': '''You are a student taking an exam to answer questions based on visualizations.
We will provide you a visualization and a list of tasks. Your goal is to read the chart and answer the question provided in the task.

* If the task asks for a value, read the chart and provide the value directly. If you have trouble reading the exact value, provide a close estimation and indicate "approximately".
* If the task asks for trend, answer it with one of "increasing", "decreasing" if the general trend point to the direcrtion, otherwise provide "unclear".
* Provide a brief reasoning of how you come up with your answer in "reasoning" part.
* If you cannot answer a question, provide "I don't know" as the answer, try not to provide a wrong answer.

Answer your question based on [Chart] and [Tasks].
The output json should have the format [{"reasoning": ..., "anwer": ...}, ....], including the answer for each questions.
'''}]}

test_tasks = [
    "Identify the average CO2 emissions of Australia.",  
    "Compare the average CO2 emissions of Australia and another entity of your choice.",  
    "Check if Australia's average CO2 emissions are above the overall average.",  
    "Point out the entity with the least average CO2 emissions.",  
    "Find out the entity with the highest average CO2 emissions.",  
    "Verify if Australia's average CO2 emissions are in the top 5 among all entities.",  
    "Observe the trend of CO2 emissions for Australia.",  
]

test_inputs = {
    'role': 'user',
    'content': [ {
        'type': 'text',
        'text': '''[Chart]
'''},
        {
            'type': 'image_url',
            'image_url': {
                "url": convert_image_to_url(os.path.join(example_image_dir, 'chart-average-co2-emission.png')),
                "detail": "low"
            }
        },
        {
            'type': 'text',
            'text': f'''[Tasks]
{json.dumps(test_tasks)}
            [Output]\n'''
        }
    ]
}


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

class ChartVisualReasoningAgent(object):

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def run(self, image_fpath, tasks, n=1, detail='auto', reading_guidance=None):

        png_url_data = convert_image_to_url(image_fpath)
        if reading_guidance is not None:
            text = f'''{reading_guidance}\n[Tasks]\n\n{json.dumps(tasks)}\n\n[Answers]\n'''
        else:
            text = f'''[Tasks]\n\n{json.dumps(tasks)}\n\n[Answers]\n'''
        # print("input text: ", text)
        query = {
            'role': 'user',
            'content': [ {
                    'type': 'text',
                    'text': f'''[Chart]\n'''
                }, {
                    'type': 'image_url',
                    'image_url': {
                        "url": png_url_data,
                        "detail": detail
                    }
                }, {
                    'type': 'text',
                    'text': text
                }
            ]
        }

        logger.info(">>> Chart Visual Reasoning Agent <<<\n")
        logger.info("##### [user query] #####")
        logger.info(json.dumps(tasks))

        messages = [system_message, query]

        response = self.client.chat.completions.create(
                    model=self.model, messages = messages, temperature=0.7, 
                    max_tokens=1200, top_p=0.95, n=n)

        # log = {'messages': messages, 'response': response.model_dump(mode='json') }
        log = {'messages': messages, 'response': json.dumps(response.model_dump()) }


        logger.info("##### [agent responses] #####")
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