# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:40:51 2021

@author: MBo
"""

from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

# modules = glob.glob(join(dirname(__file__), '*.py'))
# __all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

'''Added code below to dynamically import all classes of submodule.'''

# Iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):
    
    # Import the module and iterate through its attributes
    module = import_module(f'{__name__}.{module_name}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        
        # Add the class to this package's variables
        if isclass(attribute):
            globals()[attribute_name] = attribute
