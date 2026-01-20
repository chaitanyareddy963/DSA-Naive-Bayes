import unittest
from src.preprocessing.missing_handler import MissingHandler, compute_median


class TestMissingHandler(unittest.TestCase):
    def test_compute_median(self):
        self.assertEqual(compute_median([1.0, 3.0, 2.0]), 2.0)
        self.assertEqual(compute_median([1.0, 4.0, 2.0, 3.0]), 2.5)
        self.assertEqual(compute_median([]), 0.0)

    def test_median_imputation(self):
        handler = MissingHandler()
        rows = [
            ["1", "a"],
            ["5", "?"],
            ["9", "b"],
            ["?", "c"]
        ]
        numeric_idxs = [0]
        
        handler.fit(rows, numeric_idxs)
        self.assertEqual(handler.numeric_medians[0], 5.0)
        
        transformed = handler.transform_numeric(rows, numeric_idxs)
        self.assertEqual(transformed[0][0], 1.0)
        self.assertEqual(transformed[3][0], 5.0)

    def test_categorical_unknown(self):
        handler = MissingHandler()
        rows = [
            ["?"],
            ["non-missing"],
            [""]
        ]
        transformed = handler.transform_categorical(rows, [0], unknown_token="Unknown")
        self.assertEqual(transformed[0][0], "Unknown")
        self.assertEqual(transformed[1][0], "non-missing")
        self.assertEqual(transformed[2][0], "Unknown")


if __name__ == "__main__":
    unittest.main()
