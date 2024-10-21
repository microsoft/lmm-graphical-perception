import json
import keyword
import pandas as pd
import numpy as np

import base64

from pprint import pprint

import io
import copy
from PIL import Image 
import vl_convert as vlc
import tempfile
import re

def string_to_py_varname(var_str): 
    var_name = re.sub('\W|^(?=\d)','_', var_str)
    if keyword.iskeyword(var_name):
        var_name = f"__{var_name}"
    return var_name

def field_name_to_ts_variable_name(field_name):
    if field_name.strip() == "":
        return "inp"
    clean_name = re.sub('[^A-Za-z0-9]+', ' ', field_name)
    clean_name = re.sub(' +', ' ', clean_name)
    var_name = ''.join(x for x in clean_name.title() if not x.isspace())
    var_name = var_name[0].lower() + var_name[1:]
    return var_name

def infer_ts_datatype(df, name):
    if name not in df.columns:
        return "any"
        
    dtype = df[name].dtype
    if dtype == "object":  
        return "string"  
    elif dtype == "int64" or dtype == "float64":  
        return "number"  
    elif dtype == "bool":  
        return "boolean"  
    elif dtype == "datetime64":  
        return "Date"  
    else:  
        return "any"  

def value_handling_func(val):
    """process values to make it comparable"""
    if isinstance(val, (int,)):
        return val
    try:
        val = float(val)
        val = np.round(val, 5)
    except:
        pass

    if isinstance(val, (list,)):
        return str(val)

    return val

def table_hash(table):
    """hash a table, mostly for the purpose of comparison"""
    if len(table) == 0:
        return hash(table)
    schema = sorted(list(table[0].keys()))
    frozen_table = tuple(sorted([tuple([hash(value_handling_func(r[key])) for key in schema]) for r in table]))
    return hash(frozen_table)

def extract_code_from_gpt_response(code_raw, language):
    """search for matches and then look for pairs of ```...``` to extract code"""

    prefix_pos = [m.span()[0] for m in re.compile(f"```{language}").finditer(code_raw)]
    all_spans = [m.span() for m in re.compile("```").finditer(code_raw)]

    matches = []
    for i in range(len(all_spans) - 1):
        if all_spans[i][0] in prefix_pos and all_spans[i+1][0] not in prefix_pos:
            matches.append([all_spans[i][0], all_spans[i+1][1]])    
        
    results = []
    if len(matches) > 0:
        match = matches[0]

        for match in matches:
            code = code_raw[match[0]: match[1]]
            code = code[len(f"```{language}"): len(code) - len("```")]
            results.append(code)
            # print(code)
    # else:
    #     if language == "json":
    #         # special treatment for json objects
    #         results = extract_json_objects(code_raw)
    #     else:
    #         code = code_raw
        
    return results

def find_matching_bracket(text, start_index, bracket_type='curly'):  
    """Find the index of the matching closing bracket for JSON objects or arrays."""  
    if bracket_type == 'curly':  
        open_bracket, close_bracket = '{', '}'  
    elif bracket_type == 'square':  
        open_bracket, close_bracket = '[', ']'  
    else:  
        raise ValueError("Invalid bracket_type. Use 'curly' or 'square'.")  
  
    stack = []  
    for index in range(start_index, len(text)):  
        char = text[index]  
        if char == open_bracket:  
            stack.append(char)  
        elif char == close_bracket:  
            if not stack:  
                return -1  
            stack.pop()  
            if not stack:  
                return index  
    return -1  
  
def extract_json_objects(text):  
    """Extracts JSON objects and arrays from a text string.  
    Returns a list of parsed JSON objects and arrays.  
    """  
    json_objects = []  
    start_index = 0  
    while True:  
        # Search for the start of a JSON object or array  
        object_start = text.find('{', start_index)  
        array_start = text.find('[', start_index)  
          
        # Find the earliest JSON structure start  
        if object_start == -1 and array_start == -1:  
            break  
        elif object_start == -1:  
            start_index = array_start  
            bracket_type = 'square'  
        elif array_start == -1:  
            start_index = object_start  
            bracket_type = 'curly'  
        else:  
            start_index = min(object_start, array_start)  
            bracket_type = 'square' if start_index == array_start else 'curly'  
          
        # Find the matching closing bracket  
        end_index = find_matching_bracket(text, start_index, bracket_type)  
        if end_index == -1:  
            break  
          
        json_str = text[start_index:end_index + 1]  
        try:  
            json_obj = json.loads(json_str)  
            json_objects.append(json_obj)  
        except ValueError:  
            pass  
  
        start_index = end_index + 1  
  
    return json_objects  

# def extract_json_objects(text):  
#     """  
#     Extracts JSON objects from a text string.  
#     Returns a list of parsed JSON objects.  
#     """  
#     json_objects = []  
#     start_index = 0  
#     while True:  
#         start_index = text.find('{', start_index)  
#         if start_index == -1:  
#             break  
#         end_index = find_matching_brace(text, start_index)  
#         if end_index == -1:  
#             break  
#         json_str = text[start_index:end_index + 1]
#         try:  
#             json_obj = json.loads(json_str)  
#             json_objects.append(json_obj)  
#         except ValueError:  
#             pass  
#         start_index = end_index + 1  
#     return json_objects  

# def find_matching_brace(text, start_index):  
#     """  
#     Given a text string and an index of an opening brace,  
#     finds the index of the matching closing brace.  
#     Returns -1 if no matching closing brace is found.  
#     """  
#     stack = []  
#     for i in range(start_index, len(text)):  
#         if text[i] == '{':  
#             stack.append('{')  
#         elif text[i] == '}':  
#             if not stack:  
#                 return -1  
#             stack.pop()  
#             if not stack:  
#                 return i  
#     return -1  

def insert_candidates(code, table, dialog, candidate_groups):
    """ Try to insert a candidate into existing candidate groups
    Args:
        code: code candidate
        table: json records table
        candidate_groups: current candidate group
    Returns:
        a boolean flag incidate whether new_group_created
    """
    table_headers = sorted(table[0].keys())
    t_hash = table_hash([{c: r[c] for c in table_headers} for r in table])
    if t_hash in candidate_groups:
        candidate_groups[t_hash].append({"code": code, "content": table, "dialog": dialog})
        new_group_created = False
    else:
        candidate_groups[t_hash] = [{"code": code, "content": table, "dialog": dialog}]
        new_group_created = True
    return new_group_created

def dedup_data_transform_candidates(candidates):
    """each candidate is a tuple of {status: ..., code: ..., data: ..., dialog: ...},
    this function extracts candidates that are 'ok', and removes uncessary duplicates"""
    candidate_groups = {}
    for candidate in candidates:
        insert_candidates(candidate["code"], candidate['data'], candidate['dialog'], candidate_groups)
    return [items[0] for _, items in candidate_groups.items()]


def get_field_summary(field_name, df):
    try:
        values = sorted([x for x in list(set(df[field_name].values)) if x != None])
    except:
        values = [x for x in list(set(df[field_name].values)) if x != None]

    val_sample = ""

    sample_size = 7

    if len(values) <= sample_size:
        val_sample = values
    else:
        val_sample = values[:int(sample_size / 2)] + ["..."] + values[-(sample_size - int(sample_size / 2)):]

    val_str = ', '.join([str(s) if ',' not in str(s) else f'"{str(s)}"' for s in val_sample])

    return f"{field_name} -- type: {df[field_name].dtype}, values: {val_str}"


def generate_data_summary(input_tables, include_data_samples=True):
    
    input_table_names = [f'{string_to_py_varname(t["name"])}' for t in input_tables]

    data_samples = [t['rows'][:5] for t in input_tables]

    field_summaries = []
    for input_data in input_tables:
        df = pd.DataFrame(input_data['rows'])
        s = '\n\t'.join([get_field_summary(fname, df)  for fname in list(df.columns.values)])
        field_summaries.append(s)

    table_field_summaries = [f'table_{i} ({input_table_names[i]}) fields:\n\t{s}' for i, s in enumerate(field_summaries)]
    
    if include_data_samples:
        table_sample_strings = [f'table_{i} ({input_table_names[i]}) sample:\n\n```\n{pd.DataFrame(data_sample).to_csv(sep="|")}......\n```' for i, data_sample in enumerate(data_samples)]
    else:
        table_sample_strings = ['' for i, data_sample in enumerate(data_samples)]

    table_summary = "\n\n".join([f'{field_summary}\n\n{sample_str}' for field_summary, sample_str in zip(table_field_summaries, table_sample_strings)])

    data_summary = f'''Here are our datasets, here are their field summaries and samples:

{table_summary}
'''

    return data_summary

def convert_image_to_url(file_path, max_dimension=None):
    # utilize PIL to process image resizing

    img = Image.open(file_path)
    width = img.size[0]
    height = img.size[1]
    ext = file_path.split('.')[-1]

    if max_dimension != None and (width > max_dimension or height > max_dimension):
        wpercent = max_dimension / float(width)
        hpercent = max_dimension / float(height)
        percent = min(wpercent, hpercent)

        new_width = int((float(width) * float(percent)))
        new_height = int((float(height) * float(percent)))
    
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    bytes_output = io.BytesIO()
    img.save(bytes_output, format=ext)
    binary_fc = bytes_output.getvalue()
        
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    
    dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
    return dataurl

def save_vl_png(vl_spec, data_dict, save_file=None):
    vl_spec_copy = copy.deepcopy(vl_spec)
    data = copy.deepcopy(data_dict)
    # import pdb; pdb.set_trace()
    vl_spec_copy['data'] = {"values": data}
                       
    # create a temp file to save vega lite plot
    png_data = vlc.vegalite_to_png(vl_spec=vl_spec_copy, scale=1, ppi=144)

    if save_file is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.png')
        save_file = tmp.name
    
    with open(save_file, 'wb') as f:
        f.write(png_data)

    return save_file


# def convert_image_to_url(file_path):

#     binary_fc  = open(file_path, 'rb').read()  # fc aka file_content
#     base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    
#     ext     = file_path.split('.')[-1]
#     dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
#     return dataurl