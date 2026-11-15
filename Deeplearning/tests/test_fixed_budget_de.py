import unittest
import numpy as np
from scipy.optimize import differential_evolution
from Deeplearning.fixed_budget_de import fixed_budget_de


class FixedBudgetDETests(unittest.TestCase):
    def test_constant_objective_still_uses_whole_budget(self):
        calls=[];rng=np.random.default_rng(0);initial=rng.uniform(-1,1,1)
        result=fixed_budget_de(lambda x: calls.append(x.copy()) or 0.,1,rng,initial,
                               budget=64,population_size=8)
        self.assertEqual(len(calls),64)
        self.assertEqual(result['nfev'],64)
        self.assertIsNotNone(result['convergence_first_observed_at'])

    def test_same_candidates_as_public_scipy_solver_before_convergence(self):
        actual=[];expected=[];rng=np.random.default_rng(2);initial=rng.uniform(-1,1,2)
        fixed_budget_de(lambda x: actual.append(x.copy()) or float(np.sum(x*x)),2,rng,initial,
                        budget=64,population_size=8)
        rng=np.random.default_rng(2);initial=rng.uniform(-1,1,2)
        population=rng.uniform(-1,1,(8,2));population[0]=initial
        differential_evolution(lambda x: expected.append(x.copy()) or float(np.sum(x*x)),
            [(-1,1)]*2,init=population,maxiter=7,polish=False,tol=0,atol=0,
            mutation=(.5,1),recombination=.7,rng=rng)
        np.testing.assert_array_equal(actual,expected)
