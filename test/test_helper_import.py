import sys
import os

# Ensure src directory is in path
sys.path.append(os.path.abspath("src"))

import helper_functions

def test_helper_functions_import():
    assert helper_functions is not None
