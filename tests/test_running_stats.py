import unittest
import math
from src.custom_ds.running_stats import RunningStats


class TestRunningStats(unittest.TestCase):
    def test_single_value(self):
        rs = RunningStats()
        rs.add(10.0)
        self.assertEqual(rs.n, 1)
        self.assertEqual(rs.mean, 10.0)
        self.assertEqual(rs.variance(), 0.0)
        self.assertEqual(rs.std_dev(), 0.0)

    def test_two_values(self):
        rs = RunningStats()
        rs.add(10.0)
        rs.add(20.0)
        self.assertEqual(rs.n, 2)
        self.assertEqual(rs.mean, 15.0)
        self.assertEqual(rs.variance(), 50.0)
        self.assertTrue(math.isclose(rs.std_dev(), math.sqrt(50.0)))

    def test_welford_correctness(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        rs = RunningStats()
        for x in data:
            rs.add(x)
            
        self.assertEqual(rs.n, 5)
        self.assertEqual(rs.mean, 3.0)
        self.assertEqual(rs.variance(), 2.5)

    def test_empty(self):
        rs = RunningStats()
        self.assertEqual(rs.variance(), 0.0)
        self.assertEqual(rs.std_dev(), 0.0)


if __name__ == "__main__":
    unittest.main()
