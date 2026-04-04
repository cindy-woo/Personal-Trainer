import json

def modify_notebook():
    with open('Pose Landmarker Notebook.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell containing `def output_joints_image(IMAGE_FILE):`
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def output_joints_image(IMAGE_FILE):' in source:
                new_source = source.replace(
                    'def output_joints_image(IMAGE_FILE):',
                    'def output_joints_image(IMAGE_FILE, detector):'
                )
                new_source = new_source.replace(
                    'output_joints_image(IMAGE_FILE_1)',
                    'output_joints_image(IMAGE_FILE_1, detector)'
                )
                new_source = new_source.replace(
                    'output_joints_image(IMAGE_FILE_2)',
                    'output_joints_image(IMAGE_FILE_2, detector)'
                )
                
                # Update the cell source
                cell['source'] = [line + ('\n' if i < len(new_source.split('\n')) - 1 else '') for i, line in enumerate(new_source.split('\n'))]
                # Filter out empty strings that may have been added incorrectly by the above split logic, 
                # actually it's better to just write the lines back safely:
                lines = new_source.splitlines(True)
                cell['source'] = lines

    with open('Pose Landmarker Notebook.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    modify_notebook()
