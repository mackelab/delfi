import importlib
import os
import sys
import webbrowser

from delfi.utils.automacdoc import *
from pathlib import Path
from subprocess import call, Popen


def main(argv=None):
    argv = sys.argv if argv is None else argv

    #write_doc(argv[1], argv[2])

    modules = [
        'delfi.distribution',
        'delfi.generator',
        'delfi.inference',
        'delfi.neuralnet.NeuralNet',
        'delfi.simulator',
        'delfi.summarystats',
        'delfi.utils.viz']

    classes = []
    functions = []

    for module_name in modules:
        module = importlib.import_module(module_name, package='delfi')

        for name, obj in inspect.getmembers(module, inspect.isclass):
            cls = create_class(name, obj, ignore_prefix_function=None)
            classes.append(cls)

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            fun = create_fun(name, obj, ignore_prefix_function=None)
            functions.append(fun)

    for cls in classes:
        md_path = Path(__file__).absolute().parent / 'autodoc/{module}.md'.format(
            module=cls['module'])
        with open(md_path, 'w') as md_file:
            write_class(md_file, cls)

    for fun in functions:
        md_path = Path(__file__).absolute().parent / 'autodoc/{module}.{name}.md'.format(
            module=fun['module'], name=fun['name'])
        with open(md_path, 'w') as md_file:
            write_function(md_file, fun)

    #os.chdir(argv[2])

    #call(["mkdocs", "build", "--clean"])
    #Popen(["mkdocs", "serve"])
    #webbrowser.open("http://127.0.0.1:8000/")

if __name__ == "__main__":
    main(sys.argv)
