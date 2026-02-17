import unittest
import pandas as pd
from unittest.mock import patch

from src.data_loader import load_data_msci # Corrected import statement

class TestDataLoader(unittest.TestCase):

    @patch('src.data_loader.pd.read_csv') # Corrected patch target
    def test_load_data_msci(self, mock_read_csv):
        """
        Test that load_data_msci works correctly by MOCKING the CSV files.
        This ensures the test runs on any machine, even without the data folder.
        """
        # 1. Create fake data to simulate the CSV files
        # Fake Country Index Data (X)
        mock_country_data = pd.DataFrame(
            {'US': [0.01, 0.02], 'UK': [0.03, 0.04]}, 
            index=pd.to_datetime(['2023-01-01', '2023-01-02'])
        )
        
        # Fake World Index Data (y)
        mock_world_data = pd.DataFrame(
            {'NDDLWI': [0.05, 0.06]}, 
            index=pd.to_datetime(['2023-01-01', '2023-01-02'])
        )

        # 2. Tell the mock to return our fake data when read_csv is called
        mock_read_csv.side_effect = [mock_country_data, mock_world_data]

        # 3. Run the function (path doesn't matter now because we are mocking!)
        result = load_data_msci(path='dummy/path/')

        # 4. Verify the results
        self.assertIsNotNone(result)
        self.assertIn('return_series', result)
        self.assertIn('bm_series', result)
        
        # Check that we got our fake data back
        pd.testing.assert_frame_equal(result['return_series'], mock_country_data)
        pd.testing.assert_frame_equal(result['bm_series'], mock_world_data)
        

if __name__ == '__main__':
    unittest.main()
