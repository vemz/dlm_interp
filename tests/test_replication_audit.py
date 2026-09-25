import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/scripts'))
from audit_calendar_replication import trajectory_features


class TemporalAuditTests(unittest.TestCase):
    def test_schedule_difference_without_content_conflict(self):
        a=[[9,9],[1,9],[1,2]]
        b=[[9,9],[9,2],[1,2]]
        metrics,features=trajectory_features(a,b,mask=9,swap_step=0)
        self.assertEqual([m['state_hamming'] for m in metrics],[0,2,0])
        self.assertIsNone(features['first_committed_conflict'])
        self.assertEqual(features['first_reconvergence'],2)

    def test_conflict_timing_not_mask_timing(self):
        a=[[9,9],[1,9],[1,2]]
        b=[[9,9],[9,2],[3,2]]
        _,features=trajectory_features(a,b,mask=9,swap_step=0)
        self.assertEqual(features['first_committed_conflict'],2)
        self.assertIsNone(features['first_reconvergence'])
        self.assertEqual(features['final_hamming'],1)

    def test_malformed_state_lengths_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_features([[9,9]],[[9]],9,0)


if __name__=='__main__':
    unittest.main()
