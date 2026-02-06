import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(1, 'src')

from data_loader import load_data_msci

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # This method is run before each test
        self.data_path = os.path.join(os.getcwd(), 'data/')

    def test_load_data_msci(self):
        # Test if data can be loaded without errors
        try:
            data = load_data_msci(self.data_path)
            self.assertIsNotNone(data)
            self.assertIsInstance(data, dict)
            self.assertIn('return_series', data)
            self.assertIn('bm_series', data)
            self.assertIsInstance(data['return_series'], pd.DataFrame)
            self.assertIsInstance(data['bm_series'], pd.DataFrame)
            self.assertFalse(data['return_series'].empty)
            self.assertFalse(data['bm_series'].empty)
            print("\nSuccessfully loaded MSCI data.")
        except Exception as e:
            self.fail(f"load_data_msci failed with an error: {e}")

if __name__ == '__main__':
    unittest.main()
