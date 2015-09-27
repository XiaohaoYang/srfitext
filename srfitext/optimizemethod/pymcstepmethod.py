import pymc3
from numpy.linalg import cholesky

from pymc3.core import *
from pymc3.step_methods.quadpotential import quad_potential

from pymc3.step_methods.arraystep import *
from numpy.random import normal, standard_cauchy, standard_exponential, poisson, random
from numpy import round, exp, copy, where
import theano

from pymc3.theanof import make_shared_replacements, join_nonshared_inputs, CallableTensor
from pymc3.step_methods import NormalProposal , CauchyProposal, LaplaceProposal, PoissonProposal, MultivariateNormalProposal


class MetropolisExt(pymc3.arraystep.ArrayStepShared):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    S : standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to quad_potential.
    scaling : scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune : bool
        Flag for tuning. Defaults to True.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    default_blocked = False

    def __init__(self, vars=None, S=None, proposal_dist=NormalProposal, scaling=1.,
                 tune=True, tune_interval=1000, model=None, recipe=None, recipescale=0.001, **kwargs):

        model = modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = inputvars(vars)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))
        self.proposal_dist = proposal_dist(S)
        self.scaling = np.atleast_1d(scaling)
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.recipe_vals = np.zeros(tune_interval)
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.array([v.dtype in discrete_types for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        shared = make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        
        self.vars = vars
        self.shared = shared
        
        varlist = []
        tempdict = {}
        for k, v in self.shared.items():
            tempdict[k.name] = v
        for name in recipe.names:
            varlist.append(tempdict.get(name, None))
        self.varlist = varlist
        self.recipe = recipe
        self.recipescale = recipescale
        
        super(MetropolisExt, self).__init__(vars, shared)
    
    def astep(self, q0):

        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            
            rstd = np.std(self.recipe_vals)
            self.recipescale = tune_r(self.accepted / float(self.tune_interval)) / rstd
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0
            self.recipe_vals *= 0

        delta = self.proposal_dist() * self.scaling
        if self.any_discrete:
            if self.all_discrete:
                delta = round(delta, 0).astype(int)
                q0 = q0.astype(int)
                q = (q0 + delta).astype(int)
            else:
                delta[self.discrete] = round(delta[self.discrete], 0).astype(int)
                q = q0 + delta
        else:
            q = q0 + delta
            
        # generate p
        p = np.array([x.get_value() if x else q[0] for x in self.varlist])
        p0 = np.array([x if x != q[0] else q0[0] for x in p])
        r0 = self.recipe.scalarResidual(p0)
        r1 = self.recipe.scalarResidual(p)
        delta = (r1 - r0) * self.recipescale
        q_new = metrop_select(-delta + self.delta_logp(q, q0), q, q0)
        
        self.steps_until_tune -= 1
        
        if q_new is q:
            self.accepted += 1
            self.recipe_vals[self.steps_until_tune] = r1
        else:
            self.recipe_vals[self.steps_until_tune] = r0
        return q_new

def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """

    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale

def tune_r(acc_rate):
    
    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale = 0.5
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale = 0.75
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale = 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale = 2.0
    elif acc_rate > 0.75:
        # increase by double
        scale = 1.5
    elif acc_rate > 0.5:
        # increase by ten percent
        scale = 1.1
    else:
        scale = 1.0

    return scale

def delta_logp(logp, vars, shared):
    [logp0], inarray0 = join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f
