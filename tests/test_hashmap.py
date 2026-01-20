import unittest
from src.custom_ds.open_addressing_hash_map import OpenAddressingHashMap


class TestOpenAddressingHashMap(unittest.TestCase):
    def test_put_get_basic(self):
        hm = OpenAddressingHashMap(capacity=8)
        hm.put("key1", 100)
        self.assertEqual(hm.get("key1"), 100)
        self.assertIsNone(hm.get("key2"))
        self.assertEqual(hm.size, 1)

    def test_collision_handling(self):
        hm = OpenAddressingHashMap(capacity=4)
        hm.put("a", 1)
        hm.put("b", 2)
        hm.put("c", 3)
        
        self.assertEqual(hm.get("a"), 1)
        self.assertEqual(hm.get("b"), 2)
        self.assertEqual(hm.get("c"), 3)

    def test_resize_correctness(self):
        hm = OpenAddressingHashMap(capacity=4, max_load_factor=0.5)
        hm.put("a", 1)
        hm.put("b", 2)
        initial_cap = len(hm._keys)
        self.assertEqual(initial_cap, 4)
        
        hm.put("c", 3)
        new_cap = len(hm._keys)
        self.assertGreater(new_cap, initial_cap)
        self.assertEqual(new_cap, 8)
        
        self.assertEqual(hm.get("a"), 1)
        self.assertEqual(hm.get("b"), 2)
        self.assertEqual(hm.get("c"), 3)
        self.assertEqual(hm.size, 3)
        self.assertGreater(hm.resizes, 0)

    def test_delete(self):
        hm = OpenAddressingHashMap()
        hm.put("a", 1)
        hm.put("b", 2)
        
        self.assertTrue(hm.delete("a"))
        self.assertIsNone(hm.get("a"))
        self.assertFalse(hm.contains("a"))
        self.assertEqual(hm.size, 1)
        
        self.assertFalse(hm.delete("nonexistent"))
        
        hm.put("c", 3)
        self.assertEqual(hm.get("c"), 3)

    def test_increment(self):
        hm = OpenAddressingHashMap()
        hm.increment("count", 1)
        self.assertEqual(hm.get("count"), 1)
        hm.increment("count", 5)
        self.assertEqual(hm.get("count"), 6)
        
    def test_iterators(self):
        hm = OpenAddressingHashMap()
        data = {"a": 1, "b": 2, "c": 3}
        for k, v in data.items():
            hm.put(k, v)
            
        keys = set(hm.keys())
        self.assertEqual(keys, {"a", "b", "c"})
        
        values = set(hm.values())
        self.assertEqual(values, {1, 2, 3})
        
        items = set(hm.items())
        self.assertEqual(items, {("a", 1), ("b", 2), ("c", 3)})


if __name__ == "__main__":
    unittest.main()
