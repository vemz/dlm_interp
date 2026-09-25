import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/scripts'))
from audit_calendar_advantage import ceiling, crossfit, fit_univariate, predict, holm


class AdvantageAuditTests(unittest.TestCase):
    def test_oracle_uses_union_not_sum(self):
        result=ceiling([1,1,0,0],[1,0,1,0])
        self.assertEqual(result['oracle_correct'],3)
        self.assertEqual(result['margin_pp'],25.)

    def test_held_out_labels_do_not_change_own_predictions(self):
        x=np.column_stack([np.linspace(-2,2,50)]*4)
        labels=(np.arange(50)%3 == 0).astype(int)
        folds=np.arange(50)%5
        a,_,_=crossfit(x,labels,folds)
        changed=labels.copy();changed[folds==0]=1-changed[folds==0]
        b,_,_=crossfit(x,changed,folds)
        np.testing.assert_array_equal(a[folds==0],b[folds==0])

    def test_logistic_solver_reaches_penalized_stationary_point(self):
        rng=np.random.default_rng(22)
        x=rng.normal(size=(80,4)); y=(rng.random(80)<.45).astype(int)
        fit=fit_univariate(x,y);mean,scale,beta=fit
        residual=predict(fit,x)-y[:,None]
        np.testing.assert_allclose(residual.sum(0),0,atol=1e-7)
        np.testing.assert_allclose((residual*((x-mean)/scale)).sum(0)+beta[:,1],0,atol=1e-7)

    def test_crossfit_recovers_known_direction_without_future_gate(self):
        x=np.column_stack([np.linspace(-3,3,100)]*4)
        labels=(x[:,0]>0).astype(int)
        labels[::7]=-1  # Concordant rows are never given to the training target.
        scores,_,_=crossfit(x,labels,np.arange(100)%5)
        self.assertEqual(scores.shape,(100,4))
        event=labels>=0
        self.assertGreater(((scores[event,0]>=.5)==labels[event]).mean(),.95)

    def test_holm_controls_feature_family(self):
        np.testing.assert_allclose(holm(np.array([.01,.04,.03,.5])),[.04,.09,.09,.5])


if __name__=='__main__':unittest.main()
