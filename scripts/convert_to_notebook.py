"""
Script to convert Python script to Jupyter Notebook
"""
import json
import nbformat as nbf
from pathlib import Path

# Define paths
script_path = Path('../notebooks/1_Data_Exploration_and_Preprocessing/notebook_content.py')
notebook_path = Path('../notebooks/1_Data_Exploration_and_Preprocessing/Data_Exploration_and_Preprocessing.ipynb')

# Read the Python script
with open(script_path, 'r') as f:
    script_content = f.read()

# Split the script into cells (assuming cell separation with # %%)
cells = []
current_cell = []
current_cell_type = 'code'

for line in script_content.split('\n'):
    if line.strip() == '# %%':
        if current_cell:
            cells.append(('code', '\n'.join(current_cell)))
            current_cell = []
    elif line.startswith('# '):
        if current_cell and current_cell_type == 'markdown':
            current_cell.append(line[2:])
        else:
            if current_cell:
                cells.append((current_cell_type, '\n'.join(current_cell)))
            current_cell_type = 'markdown'
            current_cell = [line[2:]]
    else:
        if current_cell_type == 'markdown' and current_cell:
            cells.append(('markdown', '\n'.join(current_cell)))
            current_cell_type = 'code'
            current_cell = []
        current_cell.append(line)

# Add the last cell
if current_cell:
    cells.append((current_cell_type, '\n'.join(current_cell)))

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
for cell_type, source in cells:
    if cell_type == 'markdown':
        nb['cells'].append(nbf.v4.new_markdown_cell(source))
    else:
        nb['cells'].append(nbf.v4.new_code_cell(source))

# Write the notebook to a file
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created at: {notebook_path}")
