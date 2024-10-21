
import pandas as pd
import dataframe_image as dfi

# chart_type2edit_func = {
#     'bar_anno': edit_bar_w_annotation,
#     'line_anno': edit_line_w_annotation,
#     'scatter_anno': edit_scatter_w_annotation,
#     'pie_anno': edit_pie_w_annotation,
#     'bar': edit_bar_wo_annotation,
#     'line': edit_line_wo_annotation,
#     'scatter': edit_scatter_wo_annotation,
#     'pie': edit_pie_wo_annotation
# }



# -----------------------------Chart Type Experiment (RQ1)---------------------------------


# Bar (With Numerical Annotation)
def edit_bar_w_annotation(source_data):
    source_data['vl_spec'].pop('mark', None)
    ori_encoding = source_data['vl_spec'].pop('encoding')
    encoding_w_text = ori_encoding.copy()
    
    # render_vl_png(input_data, vl_spec)
    if ori_encoding['y']['type'] == 'quantitative':
        encoding_w_text['text'] = ori_encoding['y']
        text_mark = {'type': 'text', 'align': 'center', 'dy': -5}
    else:
        encoding_w_text['text'] = ori_encoding['x']
        text_mark = {'type': 'text', 'align': 'left', 'dx': 5}

    source_data['vl_spec']['layer'] = [
        {
            'mark': 'bar',
            'encoding': ori_encoding
        },
        {
            'mark': text_mark,
            'encoding': encoding_w_text
        }
    ]

    return source_data

# Bar (Without Numerical Annotation)
def edit_bar_wo_annotation(source_data):
    source_data['vl_spec']['mark'] = 'bar'
    return source_data


# Line (With Numerical Annotation)
def edit_line_w_annotation(source_data):
    source_data['vl_spec'].pop('mark', None)
    ori_encoding = source_data['vl_spec'].pop('encoding')
    encoding_w_text = ori_encoding.copy()
    
    # render_vl_png(input_data, vl_spec)
    if ori_encoding['y']['type'] == 'quantitative':
        encoding_w_text['text'] = ori_encoding['y']
        text_mark = {'type': 'text', 'align': 'center', 'dy': -8}
    else:
        encoding_w_text['text'] = ori_encoding['x']
        text_mark = {'type': 'text', 'align': 'left', 'dx': 8}

    source_data['vl_spec']['layer'] = [
        {
            'mark': 'line',
            'encoding': ori_encoding
        },
        {
            'mark': text_mark,
            'encoding': encoding_w_text
        }
    ]

    return source_data

# Line (Without Numerical Annotation)
def edit_line_wo_annotation(source_data):
    source_data['vl_spec']['mark'] = 'line'
    return source_data


# Scatter (With Numerical Annotation)
def edit_scatter_w_annotation(source_data):
    source_data['vl_spec'].pop('mark', None)
    ori_encoding = source_data['vl_spec'].pop('encoding')
    encoding_w_text = ori_encoding.copy()
    
    # render_vl_png(input_data, vl_spec)
    if ori_encoding['y']['type'] == 'quantitative':
        encoding_w_text['text'] = ori_encoding['y']
        text_mark = {'type': 'text', 'align': 'center', 'dy': -8}
    else:
        encoding_w_text['text'] = ori_encoding['x']
        text_mark = {'type': 'text', 'align': 'left', 'dx': 8}

    source_data['vl_spec']['layer'] = [
        {
            'mark': 'point',
            'encoding': ori_encoding
        },
        {
            'mark': text_mark,
            'encoding': encoding_w_text
        }
    ]

    return source_data

# Scatter (Without Numerical Annotation)
def edit_scatter_wo_annotation(source_data):
    source_data['vl_spec']['mark'] = 'point'
    return source_data


# Pie (With Numerical Annotation)
def edit_pie_w_annotation(source_data):
    source_data['vl_spec']['mark'] = 'arc'

    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':
        source_data['vl_spec']['encoding']['theta'] = source_data['vl_spec']['encoding']['y']
        source_data['vl_spec']['encoding']['color'] = source_data['vl_spec']['encoding']['x']
    else:    
        source_data['vl_spec']['encoding']['theta'] = source_data['vl_spec']['encoding']['x']
        source_data['vl_spec']['encoding']['color'] = source_data['vl_spec']['encoding']['y']
    
    source_data['vl_spec']['encoding']['color']['type'] = 'nominal'
    source_data['vl_spec']['encoding'].pop('x', None)
    source_data['vl_spec']['encoding'].pop('y', None)  
    text_encoding = source_data['vl_spec']['encoding'].copy()
    text_encoding['theta']['stack'] = True
    text_encoding['text'] = source_data['vl_spec']['encoding']['theta']
    source_data['vl_spec']['layer'] = [
        {
            'mark':  {'type': 'arc', 'outerRadius': 100} 
         },
        {
            'mark': {'type': 'text', "radiusOffset": 20, "radius": 100},
            'encoding': text_encoding
        }
    ]

    return source_data


# Table
def edit_table(source_data, output_path):
    """
    export the table to a png file, save it to the output path
    """
    data_dict = source_data['data_dict']
    df = pd.DataFrame(data_dict)
    dfi.export(df, output_path, dpi=300, max_rows=99)
    return

# -----------------------------Visual Element Experiment (RQ2)---------------------------------

# Length only, position is randomly generated
def edit_unaligned_rule(source_data):
    source_data['vl_spec'].pop('mark', None)

    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        file_name = source_data['vl_spec']['encoding']['y']['field']
        # get average value
        try:
            for d in source_data['data_dict']:
                d[file_name] = float(d[file_name])
        except:
            print("error in id: ", id)
        avg_value = sum([d[file_name] for d in source_data['data_dict']]) / len(source_data['data_dict'])
        avg_value = round(avg_value, 3)
        min_value = min([d[file_name] for d in source_data['data_dict']])
        max_value = max([d[file_name] for d in source_data['data_dict']])
        if source_data['vl_spec']['encoding']['x']['type'] == 'temporal':
            max_timestamp = max([pd.Timestamp(d[source_data['vl_spec']['encoding']['x']['field']]) for d in source_data['data_dict']])
            min_timestamp = min([pd.Timestamp(d[source_data['vl_spec']['encoding']['x']['field']]) for d in source_data['data_dict']])

            time_duration = max_timestamp - min_timestamp
            gap_unit = time_duration / len(source_data['data_dict'])
            min_timestamp = min_timestamp - gap_unit
            # to string
            str_min_timestamp = min_timestamp.strftime("%Y-%m-%d")
            source_data['data_dict'] = [{source_data['vl_spec']['encoding']['x']['field']: str_min_timestamp, file_name: None, 'label': 'legend'}] + source_data['data_dict']
        else:
            source_data['data_dict'] = [{source_data['vl_spec']['encoding']['x']['field']: '', file_name: None, 'label': 'legend'}] + source_data['data_dict']

        # add legend rule transform
        source_data['vl_spec']['transform'] = [
            {"calculate": f"datum.label == 'legend' ? {min_value*0.5} : random() * {min_value}", "as": "y0"},
            {"calculate": f"datum.label == 'legend' ? {min_value*0.5+avg_value} : datum.y0 + datum['{file_name}']", "as": "y1"},
            {"calculate": f"datum.label == 'legend' ? '{avg_value}' : ''", "as": "label"}
        ]
        source_data['vl_spec']['encoding']['y']['field'] = 'y0'
        source_data['vl_spec']['encoding']['y']['scale'] = {'domain': [0, max_value+min_value]}
        source_data['vl_spec']['encoding']['y']['axis'] = {
                                                                "title": None,
                                                                "labelFontSize": 0,
                                                                "ticks": False,
                                                                "labels": False,
                                                                "domain": False
                                                            }
        source_data['vl_spec']['encoding']['y2'] = {'field': 'y1'}

        source_data['vl_spec']['layer'] = [{
                                    "mark": {'type': 'bar', 'width': 2}
                                }, {
                                    "mark": {
                                    "type": "text",
                                    "align": "left",
                                    "baseline": "middle",
                                    "dx": -30,
                                    "dy": -30
                                    },
                                    "encoding": {
                                    "text": {"field": "label", "type": "nominal"}
                                    }
                                }]
        
        source_data['vl_spec']['view'] = {'stroke': 'transparent'}

        if isinstance(source_data['vl_spec']['title'], str):
            source_data['vl_spec']['title'] = source_data['vl_spec']['title'] + f" the left most rule is the mean value: {round(avg_value, 3)}"
        else:
            source_data['vl_spec']['title'].append(f" the left most rule is the mean value: {round(avg_value, 3)}")
        
    else:  # x is value
        file_name = source_data['vl_spec']['encoding']['x']['field']
        try:
            for d in source_data['data_dict']:
                d[file_name] = float(d[file_name])
        except:
            print("error in id: ", id)

        avg_value = sum([d[file_name] for d in source_data['data_dict']]) / len(source_data['data_dict'])
        avg_value = round(avg_value, 3)
        min_value = min([d[file_name] for d in source_data['data_dict']])
        max_value = max([d[file_name] for d in source_data['data_dict']]) 

        if source_data['vl_spec']['encoding']['y']['type'] == 'temporal':

            max_timestamp = max([pd.Timestamp(d[source_data['vl_spec']['encoding']['y']['field']]) for d in source_data['data_dict']])
            min_timestamp = min([pd.Timestamp(d[source_data['vl_spec']['encoding']['y']['field']]) for d in source_data['data_dict']])

            time_duration = max_timestamp - min_timestamp
            gap_unit = time_duration / len(source_data['data_dict'])
            min_timestamp = min_timestamp - gap_unit

            # to string
            str_min_timestamp = min_timestamp.strftime("%Y-%m-%d")

            source_data['data_dict'] = [{source_data['vl_spec']['encoding']['y']['field']: str_min_timestamp, file_name: None, 'label': 'legend'}] + source_data['data_dict']

        else:
            source_data['data_dict'].insert(0, {source_data['vl_spec']['encoding']['y']['field']: '', file_name: None, 'label': 'legend'})

        # add legend rule transform
        source_data['vl_spec']['transform'] = [
            {"calculate": f"datum.label == 'legend' ? {min_value * 0.5} : random() * {min_value}", "as": "x0"},
            {"calculate": f"datum.label == 'legend' ? {min_value*0.5+avg_value} : datum.x0 + datum['{file_name}']", "as": "x1"},
            {"calculate": f"datum.label == 'legend' ? '{avg_value}' : ''", "as": "label"}
        ]
        source_data['vl_spec']['encoding']['x']['field'] = 'x0'
        source_data['vl_spec']['encoding']['x']['scale'] = {'domain': [0, max_value+min_value]}
        source_data['vl_spec']['encoding']['x']['axis'] = {
                                                                "title": None,
                                                                "labelFontSize": 0,
                                                                "ticks": False,
                                                                "labels": False,
                                                                "domain": False
                                                            }

        source_data['vl_spec']['encoding']['x2'] = {'field': 'x1'}

        source_data['vl_spec']['layer'] = [{
                                    "mark": {'type': 'bar', 'height': 2}
                                            }, {
                                    "mark": {
                                    "type": "text",
                                    "align": "left",
                                    "baseline": "middle",
                                    "dx": 0,
                                    "dy": -10
                                    },
                                    "encoding": {
                                    "text": {"field": "label", "type": "nominal"}
                                    }
                                }]
        source_data['vl_spec']['view'] = {'stroke': 'transparent'}


        if isinstance(source_data['vl_spec']['title'], str):
            source_data['vl_spec']['title'] = source_data['vl_spec']['title'] + f" the up most rule is the mean value: {round(avg_value, 3)}"
        else:
            source_data['vl_spec']['title'].append(f" the up most rule is the mean value: {round(avg_value, 3)}")
    # print("vl_spec: ", source_data['vl_spec'])
    # print("data: ", source_data['data_dict'])
    return source_data

# Color only
def edit_color(source_data):
    source_data['vl_spec']['encoding']['color'] = {}
    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        source_data['vl_spec']['encoding']['color']['field'] = source_data['vl_spec']['encoding']['y']['field']
        # remove it
        source_data['vl_spec']['encoding'].pop('y', None)
        # check if x type is temporal, change it to ordinal.
        if source_data['vl_spec']['encoding']['x']['type'] == 'temporal':
            source_data['vl_spec']['encoding']['x']['type'] = 'ordinal'
    else:
        source_data['vl_spec']['encoding']['color']['field'] = source_data['vl_spec']['encoding']['x']['field']
        # remove it
        source_data['vl_spec']['encoding'].pop('x', None)
        if source_data['vl_spec']['encoding']['y']['type'] == 'temporal':
            source_data['vl_spec']['encoding']['y']['type'] = 'ordinal'

    source_data['vl_spec']['encoding']['color']['type'] = 'quantitative'
    source_data['vl_spec']['mark'] = 'rect'
 
    return source_data

# Size only
def edit_size(source_data):
    source_data['vl_spec']['mark'] = 'point'
    
    source_data['vl_spec']['encoding']['size'] = {}
    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        source_data['vl_spec']['encoding']['size']['field'] = source_data['vl_spec']['encoding']['y']['field']
        source_data['vl_spec']['encoding']['size']['type'] = 'quantitative'
        file_name = source_data['vl_spec']['encoding']['y']['field']
        # remove it
        source_data['vl_spec']['encoding'].pop('y', None)

    else:
        source_data['vl_spec']['encoding']['size']['field'] = source_data['vl_spec']['encoding']['x']['field']
        source_data['vl_spec']['encoding']['size']['type'] = 'quantitative'
        file_name = source_data['vl_spec']['encoding']['x']['field']
        # remove it
        source_data['vl_spec']['encoding'].pop('x', None)

    for d in source_data['data_dict']:
        d[file_name] = float(d[file_name])
    avg_value = sum([d[file_name] for d in source_data['data_dict']]) / len(source_data['data_dict'])
    avg_value = round(avg_value, 3)
    min_value = min([d[file_name] for d in source_data['data_dict']])
    max_value = max([d[file_name] for d in source_data['data_dict']])

    # if the range of the data is smaller than the min or max value, set the scale to not zero (someone's rule, it's a classical rule but I can't remember who lol)
    ext = abs(max_value - min_value)
    if ext < abs(min_value) and ext < abs(max_value):
        source_data['vl_spec']['encoding']['size']['scale'] = {'zero': False}

    source_data['vl_spec']['config']['legend'] = {
      "labelFontSize": 11,
      "padding": 1,
      "symbolSize": 30,
      "symbolType": "circle"
    }

    return source_data

# Scatter Size (Position + Size)
def edit_scatter_size(source_data):
    source_data['vl_spec']['mark'] = 'point'
    # render_vl_png(input_data, vl_spec)
    source_data['vl_spec']['encoding']['size'] = {}
    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        source_data['vl_spec']['encoding']['size']['field'] = source_data['vl_spec']['encoding']['y']['field']
        source_data['vl_spec']['encoding']['size']['type'] = 'quantitative'
        file_name = source_data['vl_spec']['encoding']['y']['field']
    else:
        source_data['vl_spec']['encoding']['size']['field'] = source_data['vl_spec']['encoding']['x']['field']
        source_data['vl_spec']['encoding']['size']['type'] = 'quantitative'
        file_name = source_data['vl_spec']['encoding']['x']['field']


    for d in source_data['data_dict']:
        d[file_name] = float(d[file_name])

    avg_value = sum([d[file_name] for d in source_data['data_dict']]) / len(source_data['data_dict'])
    avg_value = round(avg_value, 3)
    min_value = min([d[file_name] for d in source_data['data_dict']])
    max_value = max([d[file_name] for d in source_data['data_dict']])

    ext = abs(max_value - min_value)
    if ext < abs(min_value) and ext < abs(max_value):
        source_data['vl_spec']['encoding']['size']['scale'] = {'zero': False}

    source_data['vl_spec']['config']['legend'] = {
      "labelFontSize": 11,
      "padding": 1,
      "symbolSize": 30,
      "symbolType": "circle"
    }

    return source_data

# Rule (Position + Length)
def edit_rule_color(source_data):
    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        bar_key = 'width'
    else:
        bar_key = 'height'

    source_data['vl_spec']['mark'] = {'type': 'bar', bar_key: 2}
    return source_data

# Bar (Position + Size + Length)
def edit_bar_color(source_data):
    source_data['vl_spec']['mark'] = 'bar'
    # render_vl_png(input_data, vl_spec)
    if source_data['vl_spec']['encoding']['y']['type'] == 'quantitative':  # y is value
        source_data['vl_spec']['encoding']['color']['field'] = source_data['vl_spec']['encoding']['y']['field']
        source_data['vl_spec']['encoding']['color']['type'] = 'quantitative'
    else:
        source_data['vl_spec']['encoding']['color']['field'] = source_data['vl_spec']['encoding']['x']['field']
        source_data['vl_spec']['encoding']['color']['type'] = 'quantitative'

    return source_data




