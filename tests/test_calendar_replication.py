import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/scripts'))
from calendar_swap import Protocol, generate_pair
from calendar_replication import generate_pair_traced, prepare
from calendar_robustness import auc
from answer_audit_v4 import extract


class ReplicationTests(unittest.TestCase):
    def test_auc_orientation_and_ties(self):
        self.assertEqual(auc([0,1],[False,True]),1)
        self.assertEqual(auc([1,0],[False,True]),0)
        self.assertEqual(auc([1,1],[False,True]),.5)
        self.assertAlmostEqual(auc([0,1,1],[False,True,False]),.75)
        self.assertIsNone(auc([0,1],[True,True]))

    def test_instrumentation_matches_frozen_generator(self):
        cfg = Protocol(gen_len=8,block_len=4,steps_per_block=4,swap_step=1,mask_id=7)
        prompt = torch.tensor([1,2])
        def forward(x):
            logits = torch.zeros((*x.shape,8))
            content = int(x[x!=7].sum()) % 6
            for i in range(x.shape[1]):
                logits[0,i,(i+content)%6] = 1 + i*.1
            return logits
        expected = generate_pair(forward,prompt,cfg,controls=True)
        actual = generate_pair_traced(forward,prompt,cfg,controls=True)
        self.assertEqual({k:v for k,v in actual.items() if k!='trace'}, expected)
        self.assertEqual(len(actual['trace']['baseline_states']),9)
        self.assertEqual(actual['trace']['baseline_states'][-1],actual['baseline_ids'])
        self.assertEqual(actual['trace']['treated_states'][-1],actual['treated_ids'])
        self.assertEqual(actual['trace']['divergence'][cfg.swap_step]['state_hamming'],0)
        self.assertEqual(actual['trace']['divergence'][cfg.swap_step+1]['state_hamming'],2)

    def test_replay_detects_nondeterminism(self):
        cfg=Protocol(gen_len=4,block_len=4,steps_per_block=4,swap_step=1,mask_id=7)
        calls=0
        def forward(x):
            nonlocal calls
            calls+=1
            logits=torch.zeros((*x.shape,8))
            logits[:,:,calls%6]=5
            return logits
        with self.assertRaisesRegex(ValueError,'replay'):
            generate_pair_traced(forward,torch.tensor([1]),cfg,controls=True)

    def test_full_schedule_accounting(self):
        cfg=Protocol(mask_id=7)
        def forward(x):
            logits=torch.zeros((*x.shape,8))
            logits[:,:,2]=4
            return logits
        row=generate_pair_traced(forward,torch.tensor([1]),cfg,controls=True)
        self.assertEqual(row['nfe_actual_pair'],124)
        self.assertEqual(row['nfe_controls'],124)
        self.assertEqual(len(row['trace']['baseline_states']),65)
        self.assertTrue(row['checks']['no_swap_replay'])
        self.assertTrue(row['checks']['independent_baseline_replay'])

    def test_plan_excludes_discovery_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source.json'; out=p/'plan.json'
            source.write_text(json.dumps(dict(question_ids=[0,2],model_revision='a',dataset_revision='b',protocol={})))
            a=prepare(source,out,dataset_size=12,n=5)
            self.assertFalse(set(a['question_ids']) & {0,2})
            self.assertEqual(a,prepare(source,out,dataset_size=12,n=5))
            with self.assertRaisesRegex(ValueError,'overwrite'):
                prepare(source,out,dataset_size=12,n=6)

    def test_frozen_parser_key_guards(self):
        self.assertEqual(extract('#### -$110','What is the price in dollars?')['answer'],'-110')
        self.assertIsNone(extract('#### <4>')['answer'])
        self.assertIsNone(extract('5. Determine the number of items they have.')['answer'])
        self.assertEqual(extract('Therefore, there are 5 items.\n\n#### <answer>')['answer'],'5')


if __name__ == '__main__':
    unittest.main()
