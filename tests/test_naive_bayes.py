import unittest
from src.nb_model.naive_bayes import HybridNaiveBayes


class TestNaiveBayes(unittest.TestCase):
    def test_fit_predict_basic(self):
        X_num = [[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]]
        X_cat = [[0], [0], [1], [1]]
        y = [0, 0, 1, 1]
        cat_cardinalities = [2]
        
        model = HybridNaiveBayes(alpha=1.0)
        model.fit(X_num, X_cat, y, cat_cardinalities, num_classes=2)
        
        p1 = model.predict_one([1.5, 1.5], [0])
        self.assertEqual(p1, 0)
        
        p2 = model.predict_one([10.5, 10.5], [1])
        self.assertEqual(p2, 1)

    def test_laplace_smoothing_no_zeros(self):
        X_cat = [[0], [0]]
        X_num = [[1.0], [1.0]]
        y = [0, 0]
        cat_cardinalities = [2]
        
        model = HybridNaiveBayes(alpha=1.0)
        model.fit(X_num, X_cat, y, cat_cardinalities, num_classes=1)
        
        p = model.predict_one([1.0], [1])
        self.assertEqual(p, 0)


if __name__ == "__main__":
    unittest.main()
