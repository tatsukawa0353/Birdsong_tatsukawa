"""SciPy 1.15 DE generations with an exact objective-call budget.

The public solve() exits on a constant population even when tol=atol=0.
Iterating the same immediate-update solver keeps its mutation, crossover and
selection unchanged while deliberately continuing converged populations.
"""
import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver


def fixed_budget_de(objective, dimensions, rng, initial, *, budget, population_size,
                    mutation=(.5,1.), recombination=.7):
    if budget < 2*population_size or budget % population_size:
        raise ValueError('Budget must be a multiple of population size and at least two populations')
    population=rng.uniform(-1,1,(population_size,dimensions))
    population[0]=initial
    first_converged=None
    with DifferentialEvolutionSolver(objective,[(-1,1)]*dimensions,init=population,
            maxiter=budget//population_size-1,maxfun=budget,polish=False,tol=0,atol=0,
            mutation=tuple(mutation),recombination=recombination,rng=rng,
            updating='immediate') as solver:
        generations=0
        while solver._nfev < budget:
            next(solver)
            generations+=1
            if solver.converged() and first_converged is None:
                first_converged=int(solver._nfev)
        if solver._nfev != budget:
            raise RuntimeError('DE objective-call budget mismatch')
        return dict(action=solver.x.tolist(),error=float(solver.population_energies[0]),
                    nfev=int(solver._nfev),nit=generations,
                    convergence_first_observed_at=first_converged,
                    convergence_stops_disabled=True,
                    message='Fixed candidate budget completed, including calls after convergence',
                    role=f'DE selected incumbent; auxiliary, not actual candidate {budget}')
