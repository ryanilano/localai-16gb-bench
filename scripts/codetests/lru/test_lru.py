# Reference test suite for the `lru` code challenge (codetests/lru/prompt.txt).
# run-codetest.sh writes the model's code to solution.py in the same dir, then runs:
#   python3 -m unittest -v test_lru
# The model passes only if every assertion here holds. The peek-vs-get recency
# distinction (test_peek_does_not_update_recency) is the main discriminator.
import unittest
from solution import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_basic_get_put(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.get("b"), 2)

    def test_missing_key_returns_none(self):
        c = LRUCache(2)
        self.assertIsNone(c.get("nope"))

    def test_update_existing_value(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("a", 99)
        self.assertEqual(c.get("a"), 99)
        self.assertEqual(len(c), 1)

    def test_eviction_is_least_recently_used(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)          # capacity exceeded -> evict "a" (LRU)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("c"), 3)

    def test_get_updates_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)   # "a" is now most-recently-used
        c.put("c", 3)                     # must evict "b", NOT "a"
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("c"), 3)

    def test_put_refreshes_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)                    # re-put "a": new value + most-recently-used
        c.put("c", 3)                     # must evict "b", NOT "a"
        self.assertEqual(c.get("a"), 10)
        self.assertIsNone(c.get("b"))

    def test_peek_returns_value(self):
        c = LRUCache(2)
        c.put("a", 1)
        self.assertEqual(c.peek("a"), 1)

    def test_peek_missing_returns_none(self):
        c = LRUCache(1)
        self.assertIsNone(c.peek("nope"))

    def test_peek_does_not_update_recency(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.peek("a"), 1)  # peek must NOT bump "a"
        c.put("c", 3)                     # "a" is still LRU -> it gets evicted
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("c"), 3)

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put("a", 1)
        c.put("b", 2)                     # evicts "a"
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)

    def test_len_tracks_size(self):
        c = LRUCache(2)
        self.assertEqual(len(c), 0)
        c.put("a", 1)
        self.assertEqual(len(c), 1)
        c.put("b", 2)
        c.put("c", 3)                     # eviction keeps len at capacity
        self.assertEqual(len(c), 2)


if __name__ == "__main__":
    unittest.main()
