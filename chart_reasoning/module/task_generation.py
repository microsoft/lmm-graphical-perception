import json
import pandas as pd

from ..utils.data_utils import *

import logging
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = '''You are a teacher to provide a problem for students to solve. The problem is about understanding data and visualizations.
We will provide you with an input data, a Vega-Lite program, and a task type that the understanding task should base on. 
You will need to design a chart understanding task contextualized in the given data and chart.

Design the task based off one of the following idioms:

• Find Anomalies: ask students to identify any anomalies within a given set of data points with respect to a given relationship or expectation. For example, which genre of movies appears to have abnormal length?
• Find Clusters: for a given set of data points, ask students to count the number of groups of similar data attribute values. For example, how many different genres are shown in the chart below?
• Find Correlation: for a given set of two data attributes, ask students to determine if there is a correlation between them. For example, is there a strong correlation between average budget and movie rating?
• Compute Derived Value: for a given set of data points, ask students to compute an aggregate value of those data points. For example, what is the sum of the budget for the action and the sci-fi movies?
• Characterize Distribution: for a given set of data points, ask students to identify the distribution of that attribute's values over the set. For example, what percentage of the movie genres have an average gross value higher than 10 million?
• Find Extremum: For given concrete conditions on data attribute values, ask students to find data points satisfying those conditions. For example, which car types have city miles per gallon ranging from 25 to 56?
• Filter: For given concrete conditions on data attribute values, ask students to find data points satisfying those conditions. For example, which car types have city miles per gallon ranging from 25 to 56?
• Order: For a given set of data points, ask students to rank them according to a specific ordinal metric. For example, which of the following options contains the correct sequence of movie genres, if you were to put them in order from largest average gross value to lowest?
• Determine Range: For a given set of data points and an attribute of interest, ask students to find the span of values within the set. For example, what is the range of car prices?
• Retrieve Value. For this task, ask students to identify values of attributes for given data points. For example, what is the value of horsepower for the cars?


You need to match the following requirements:
1. The task should be reasonable, and it should not exceed one sentence, and it should be contexualized in the given data.
2. The task should be achievable by reading the visualization without referring other tools.
3. The task should be self-contained with the given dataset, it should not require student to look up external information.
4. Each task should have a standard answer, avoid generating questions like ""compare two values of your choice."
5. Try not to repeat the verb for each instruction to maximize diversity.
6. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
7. The type of instructions should be diverse based the idioms above.
8. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.

Create a [Task] based off the [Data Summary] and [VegaLite Script] provided.   
The response should be in a json format {"reason": ..., "tasks": [{"description":..., "type": ...}, ...]}, that includes how you design the task and the actual task description. 
The task description should be very concise and umabiguous, remember, you are a teacher providing a task for students to solve.
Generate 10 tasks at once.
'''

EXAMPLE = ''' For example:

[Data Summary]

```
|Date|Location|State
0|5/12/2009|Houston, TX|TX
1|4/18/2009|McAllen, TX|TX
2|7/11/2009|Indianapolis, IN|IN
3|11/14/2009|Kansas City, MO|MO
4|3/12/2010|Chicago, IL|IL
......
```

[VegaLite Script]

```json
{"mark": "bar", "encoding": {"y": {"aggregate": "count", "title": "Count"}, "x": {"field": "Location", "type": "nominal"}}}
```

[Task]

{
  "reason": "The provided Vega-Lite script produces a bar chart that counts the number of occurrences by location. Based on this, tasks can be designed to read values, compare concrete values, and identify trends. The tasks will focus on using the bar heights to interpret the count of events at various locations, compare these counts, and observe any notable patterns.",
  "tasks": [
    {
      "description": "Determine the total number of events that occurred in Houston, TX.",
      "type": "Retrieve Value"
    },
    {
      "description": "Identify the location with the highest number of events.",
      "type": "Find Extremum"
    },
    {
      "description": "Compute the average number of events across all locations.",
      "type": "Compute Derived Value"
    },
    {
      "description": "Find any location that has a significantly lower number of events compared to others.",
      "type": "Find Anomalies"
    },
    {
      "description": "Identify if there is a correlation between the location and the number of events.",
      "type": "Find Correlation"
    },
    {
      "description": "What locations have more than 10 events?",
      "type": "Filter"
    {
      "description": "Order the locations from the highest to the lowest number of events.",
      "type": "Order"
    },
    {
      "description": "Determine the range of the number of events across all locations.",
      "type": "Determine Range"
    },
    {
      "description": "What is the percentage of locations with more than 10 events?",
      "type": "Characterize Distribution"
    },
    {
      "description": "Identify any clusters of locations with similar event counts.",
      "type": "Find Clusters"
    }
  ]
}

'''


class ChartReasoningTaskAgent(object):

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def run(self, data_dict, vl_spec, n):

        # data_sample = data_dict[:20] if len(data_dict) > 20 else data_dict
        data_sample = data_dict[:50] if len(data_dict) > 50 else data_dict
        table_sample_strings = f'sample:\n\n```\n{pd.DataFrame(data_sample).to_csv(sep="|")}......\n```'

        user_query = f"[Data Summary]\n\n{table_sample_strings}\n\n[VegaLite Script]\n\n{json.dumps(vl_spec)}\n\n[Task]"

        # print(user_query)

        messages = [{"role":"system", "content": SYSTEM_PROMPT+ "\n" + EXAMPLE},
                    {"role":"user","content": user_query}]

        ###### the part that calls open_ai
        response = self.client.chat.completions.create(
            model=self.model, messages = messages, temperature=0.7, max_tokens=1200,
            top_p=0.95, n=n, frequency_penalty=0, presence_penalty=0, stop=None)

        logger.info(">>> Chart Reasoning Task Generator Agent <<<\n")

        candidates = []
        for choice in response.choices:
            
            logger.info(">>> Chart Reasoning Task Generator Agent <<<\n")
            logger.info(choice.message.content + "\n")
            
            blocks = extract_json_objects(choice.message.content + "\n")
            
            if len(blocks) > 0:
                result = {'status': 'ok', 'content': blocks[-1]}
            else:
                result = {'status': 'error', 'content': 'unable to extract json object from response'}
            
            # individual dialog for the agent
            result['dialog'] = [*messages, {"role": choice.message.role, "content": choice.message.content}]

            candidates.append(result)

        return candidates