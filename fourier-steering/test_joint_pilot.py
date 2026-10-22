"""Small mathematical/data checks; no model weights are loaded."""
import unittest

import numpy as np

from joint_pilot import features, heldout, parse, prompt


class JointPilotChecks(unittest.TestCase):
    def setUp(self):
        pairs = [(d, m) for d in range(7) for m in range(12)]
        self.train = [p for p in pairs if not heldout(*p)]
        self.test = [p for p in pairs if heldout(*p)]

    def test_split_and_template_separation(self):
        self.assertEqual((len(self.train), len(self.test)), (63, 21))
        self.assertFalse(set(self.train) & set(self.test))
        self.assertEqual({d for d, m in self.train}, set(range(7)))
        self.assertEqual({m for d, m in self.train}, set(range(12)))
        fit_prompts = {prompt(d, m, t, o) for d, m in self.train
                       for t in [0, 1] for o in ["dm", "md"]}
        eval_prompts = {prompt(d, m, 2, o) for d in range(7)
                        for m in range(12) for o in ["dm", "md"]}
        self.assertFalse(fit_prompts & eval_prompts)

    def test_full_rank_and_unseen_pair_reconstruction(self):
        rng = np.random.default_rng(0)
        for joint, width in [(False, 9), (True, 25)]:
            X, Xt = features(self.train, joint), features(self.test, joint)
            self.assertEqual(X.shape, (63, width))
            self.assertEqual(np.linalg.matrix_rank(X), width)
            truth = rng.normal(size=(width, 3))
            fit = np.linalg.lstsq(X, X @ truth, rcond=None)[0]
            np.testing.assert_allclose(Xt @ fit, Xt @ truth, atol=1e-12)

    def test_mixed_signal_requires_joint_features(self):
        y = features(self.train, True)[:, 9]
        truth = features(self.test, True)[:, 9]
        errors = []
        for joint in [False, True]:
            fit = np.linalg.lstsq(features(self.train, joint), y, rcond=None)[0]
            errors.append(np.mean((features(self.test, joint) @ fit - truth) ** 2))
        self.assertGreater(errors[0], 0.1)
        self.assertLess(errors[1], 1e-24)

    def test_parser_needs_both_labels(self):
        self.assertEqual(parse('"Tuesday, March'), (1, 2))
        self.assertEqual(parse('"March, Tuesday'), (1, 2))
        self.assertEqual(parse("Tuesday"), (1, None))
        self.assertEqual(parse("Maybe later"), (None, None))


if __name__ == "__main__":
    unittest.main()
