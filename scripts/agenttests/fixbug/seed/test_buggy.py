import unittest

from buggy import dedupe


class TestDedupe(unittest.TestCase):
    def test_removes_duplicates(self):
        self.assertEqual(dedupe([1, 2, 2, 3, 1]), [1, 2, 3])

    def test_preserves_order(self):
        self.assertEqual(dedupe(["b", "a", "b", "c", "a"]), ["b", "a", "c"])

    def test_empty(self):
        self.assertEqual(dedupe([]), [])

    def test_all_same(self):
        self.assertEqual(dedupe(["x", "x", "x"]), ["x"])


if __name__ == "__main__":
    unittest.main()
