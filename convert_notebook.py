import nbformat
import os

def convert_notebook_to_script(notebook_filename, script_filename):
    """Convert a Jupyter Notebook (.ipynb) to a Python script (.py)."""
    if not os.path.exists(notebook_filename):
        print(f"Error: {notebook_filename} does not exist.")
        return
    
    try:
        # Load notebook content
        with open(notebook_filename, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        
        code_cells = [cell.source for cell in notebook_content.cells if cell.cell_type == 'code']
        
        # Handle empty notebook or no code cells
        if not code_cells:
            print(f"Warning: {notebook_filename} contains no code cells.")
            return
        
        with open(script_filename, 'w', encoding='utf-8') as f:
            for source in code_cells:
                f.write(source + '\n\n')
        
        print(f"Converted {notebook_filename} to {script_filename}")

# List of notebooks to convert
notebook_files = [
    "NaiveBayes_and_LSTM_model.ipynb",
    "RoBERTa_model.ipynb"
]

# Convert each notebook
for notebook in notebook_files:
    script_name = notebook.replace(".ipynb", ".py")
    convert_notebook_to_script(notebook, script_name)
