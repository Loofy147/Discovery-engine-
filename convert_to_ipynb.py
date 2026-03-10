import nbformat as nbf
import sys

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        code = f.read()

    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_code_cell(code))

    with open(ipynb_file, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])
