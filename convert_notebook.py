import nbformat

# Define file names
notebook_filename = "Data_sampling_with_preprocessing_oversample_v2.ipynb"
script_filename = "Data_sampling_with_preprocessing_oversample_v2.py"

# Load the notebook
with open(notebook_filename, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Extract code cells and save to .py file
with open(script_filename, 'w', encoding='utf-8') as f:
    for cell in notebook_content.cells:
        if cell.cell_type == 'code':
            f.write(cell.source + '\n\n')

print(f"Converted {notebook_filename} to {script_filename}")