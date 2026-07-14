import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from synthetic_velocity_integration_benchmark import REGULARIZATIONS, build_cases


class Overnight125BenchmarkTest(unittest.TestCase):
    def test_complete_factorial(self) -> None:
        cases = build_cases("overnight_125")

        self.assertEqual(len(cases), 18)
        self.assertEqual(len(cases) * len(REGULARIZATIONS), 36)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertEqual(
            {case.geometry for case in cases},
            {"sparse_grid", "sparse_23", "sparse_11"},
        )
        self.assertEqual({case.speed_km_s for case in cases}, {0.36, 2.0})
        self.assertEqual({case.integration_time_sec for case in cases}, {60, 120, 300})
        self.assertEqual({case.seed for case in cases}, {0})
        self.assertTrue(all(case.motion == "left_right" for case in cases))


if __name__ == "__main__":
    unittest.main()
