import sys

import plotly.graph_objects as go
import random
import numpy as np

# Init

# file_name = 'mask-Cora.txt'
file_name = 'mask-CiteSeer.txt'

# benchmark_acc = 0.7154
benchmark_acc = 0.8373
data = []
with open(file_name, 'r') as f:
    # with open('./__resources__/' + file_name, 'r') as f:
    while True:
        fragment = f.readline()
        if not fragment:
            break
        else:
            fragment = fragment.split('\t')
            data.append([float(fragment[-1]), float(fragment[0])])

random.shuffle(data)
data.sort(key=lambda x: x[1])
input_data = []
color = []
maskrates = []
for eledata in data:
    input_data.append(eledata[0])
    color.append(eledata[0] - benchmark_acc)
    maskrates.append(eledata[1])

max_col = max(color)
min_col = -min(color)
for i in range(len(color)):
    if color[i] > 0:
        color[i] = color[i] / max_col
    elif color[i] < 0:
        color[i] = color[i] / min_col

maskrates_ori = maskrates.copy()
mask_rate = str(maskrates[0])
same_num = 1
start = 0
end = 0
for i in range(len(maskrates) - 1):
    if maskrates[i] == maskrates[i + 1]:
        same_num += 1
        maskrates[i] = ''
    else:
        same_num = 1
        end = i + 1
        maskrates[i] = ''
        maskrates[int((start + end) / 2)] = mask_rate
        start = i + 1
        mask_rate = str(maskrates[i + 1])
end = len(maskrates)
maskrates[int((start + end) / 2)] = mask_rate

axis_template = dict(
    showgrid=True,
    zeroline=False,
    nticks=10,
    showline=True,
    title='X axis',
    mirror='all',
    zerolinecolor='#FF0000'
)
layout = go.Layout(
    xaxis=axis_template,
    yaxis=axis_template
)

fig = go.Figure(data=go.Scatter(
    # x=maskrates,
    y=input_data,
    mode='markers',
    marker=dict(
        size=10,
        color=color,
        colorscale='Plasma',  # one of plotly colorscales
        showscale=True,
        line_color=color,
        line_colorscale="bluered",
        line_width=2
    )
),
    layout=layout)

fig.show()

