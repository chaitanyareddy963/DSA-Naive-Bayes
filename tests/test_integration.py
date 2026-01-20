import unittest
import os
import subprocess
import json
import shutil
import sys
import tempfile


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.synthetic_data = os.path.join(self.test_dir, "test_data.csv")
        content = (
            "age,workclass,education,income\n"
            "25,Private,Bachelors,<=50K\n"
            "30,Private,Masters,>50K\n"
            "?,Local-gov,PhD,>50K\n"
            "45,Self-emp,HS-grad,<=50K\n"
            "20,?,Some-college,<=50K\n"
        )
        with open(self.synthetic_data, "w", encoding="utf-8") as f:
            f.write(content)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_run(self):
        output_dir = os.path.join(self.test_dir, "results")
        
        cmd = [
            sys.executable, "main.py",
            "--data-path", self.synthetic_data,
            "--output-dir", output_dir,
            "--test-ratio", "0.4",
            "--seed", "42",
            "--force-numeric", "age",
            "--no-plots"
        ]
        
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"Process failed: {result.stderr}")
        
        metrics_file = os.path.join(output_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_file))
        
        with open(metrics_file) as f:
            data = json.load(f)
            
        self.assertIn("metrics", data)
        self.assertIn("hashmap_stats", data)
        
        metrics = data["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        
        stats = data["hashmap_stats"]
        self.assertGreaterEqual(stats["size"], 0)

    def test_cross_validation_run(self):
        output_dir = os.path.join(self.test_dir, "results_cv")
        cmd = [
            sys.executable, "main.py",
            "--data-path", self.synthetic_data,
            "--cv", "3",
            "--output-dir", output_dir,
            "--no-plots"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Average CV Results", result.stdout)

    def test_uci_split_flag(self):
        data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        with open(os.path.join(data_dir, "adult.data"), "w") as f:
            f.write("age,income\n20,<=50K")
        with open(os.path.join(data_dir, "adult.test"), "w") as f:
            f.write("age,income\n25,>50K")
            
        output_dir = os.path.join(self.test_dir, "results_uci")
        
        cmd = [
            sys.executable, "main.py",
            "--data-path", os.path.join(data_dir, "any.csv"),
            "--use-uci-split",
            "--output-dir", output_dir,
            "--no-plots"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"UCI split run failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
