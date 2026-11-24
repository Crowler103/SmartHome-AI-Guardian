import unittest
import numpy as np
from main import simulate, features, evaluate


class GuardianTests(unittest.TestCase):
    def test_simulation(self):
        hours, water = simulate(2, 42)
        self.assertEqual(len(water), 48)
        self.assertTrue(np.all(water >= 0))
        self.assertEqual(features(hours, water).shape, (48, 5))

    def test_features_do_not_use_future(self):
        hours, water = simulate(2, 42)
        first = features(hours, water)
        water[20:] += 100
        np.testing.assert_allclose(first[:20], features(hours, water)[:20])

    def test_demo(self):
        hours, water, truth, predicted, scores = evaluate()
        self.assertEqual(len(hours), 336)
        self.assertEqual(truth.sum(), 11)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all(np.isin(predicted, [0, 1])))
        self.assertGreater(np.sum(truth * predicted), 0)


if __name__ == '__main__':
    unittest.main()
