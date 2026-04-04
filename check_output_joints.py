import json

with open('Pose Landmarker Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def output_joints_image' in source:
            print(source)
