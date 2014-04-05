import numpy as np
import scipy as sp
import random
import os
import itertools
from functools import partial
import tables as tb

import pymc
import theano.tensor as T
from theano import function

from pymc import step_methods
from pymc.progressbar import progress_bar
from pymc.trace import NpTrace, MultiTrace
from pymc.core import modelcontext, Point

##########################################################
# sampling
#########################################################

def sample(draws, step, start=None, trace=None, tune=None, progressbar=True, model=None,
           random_seed=None, callback=None, recordinterval=1, **kwargs):

    progress = progress_bar(draws)
    try:
        for i, trace in enumerate(iter_sample(draws, step,
                                              start=start,
                                              trace=trace,
                                              tune=tune,
                                              model=model,
                                              random_seed=random_seed,
                                              recordinterval=recordinterval)):
            if progressbar:
                progress.update(i)
                if callback != None:
                    callback()
    except KeyboardInterrupt:
        pass
    return trace

def iter_sample(draws, step, start=None, trace=None, tune=None, model=None, random_seed=None, recordinterval=1):
    model = modelcontext(model)
    draws = int(draws)
    np.random.seed(random_seed)

    if start is None:
        start = {}

    if isinstance(trace, TbTrace) and len(trace) > 0:
        trace_point = trace.point(-1)
        trace_point.update(start)
        start = trace_point

    else:
        test_point = model.test_point.copy()
        test_point.update(start)
        start = test_point

        if not isinstance(trace, TbTrace):
            if trace is None:
                trace = model.unobserved_RVs
            # trace = NpTrace(trace)
            trace = TbTrace(trace)

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    for i in xrange(draws):
        if (i == tune):
            step = stop_tuning(step)
        point = step.step(point)
        if i % recordinterval == 0:
            trace.record(point)
        yield trace

##########################################################
# currently not supported
#########################################################

def argsample(args):
    """ defined at top level so it can be pickled"""
    return sample(*args)

def psample(draws, step, start=None, trace=None, tune=None, progressbar=True,
            model=None, threads=None, random_seeds=None, recordinterval=1):

    model = modelcontext(model)

    if not threads:
        threads = max(mp.cpu_count() - 2, 1)

    if start is None:
        start = {}

    if isinstance(start, dict):
        start = threads * [start]

    if trace is None:
        trace = model.vars

    if type(trace) is MultiTrace:
        mtrace = trace
    else:
        mtrace = MultiTrace(threads, trace)

    p = mp.Pool(threads)

    if random_seeds is None:
        random_seeds = [None] * threads
    pbars = [progressbar] + [False] * (threads - 1)

    argset = zip([draws] * threads, [step] * threads, start, mtrace.traces,
                 [tune] * threads, pbars, [model] * threads, random_seeds,
                 [recordinterval] * threads)

    traces = p.map(argsample, argset)
    p.close()

    return MultiTrace(traces)

##########################################################
# trace in pytables
#########################################################

class TbTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, vars, filename=None, expectedrows=50000, loadfromfile=False, **kwargs):
        vars = list(vars)
        self.vars = vars
        self.expectedrows = expectedrows
        model = vars[0].model
        self.f = model.fastfn(vars)
        self.varnames = list(map(str, vars))

        if filename == None:
            filename = 'mcmc.h5'
        if loadfromfile:
            self.loadFromFile(filename)
        else:
            self.createDB(filename)
        return

    def createDB(self, filename):
        self.dbfile = tb.openFile(filename, mode="w", title="MCMC Database")
        self.dbroot = self.dbfile.root
        filters = tb.Filters(complevel=5, complib='zlib')
        self.data = self.dbfile.createGroup('/', 'data', 'data', filters)
        group = self.data

        self.samples = {}
        h5f = self.dbfile
        for var in self.vars:
            vname = str(var)
            vshape = var.dshape
            vsize = var.dsize
            if vshape == () and vsize == 1:
                rv = h5f.createEArray(group, vname, tb.Float64Atom(), (0,), expectedrows=self.expectedrows)
            else:
                shape = (0,) + vshape
                rv = h5f.createEArray(group, vname, tb.Float64Atom(), shape, expectedrows=self.expectedrows)
            self.samples[vname] = rv
            # self.samples[vname] = getattr(self.data, vname)
        h5f.flush()
        return


    def loadFromFile(self, filename):
        if os.path.exists(filename):
            self.dbfile = tb.openFile(filename, mode="r+", title="MCMC Database")
            self.dbroot = self.dbfile.root
            self.data = self.dbroot.data
            group = self.data
            self.samples = {}
            for var in self.vars:
                vname = str(var)
                self.samples[vname] = getattr(group, vname)
            self.dbfile.flush()
        return

    def record(self, point):
        """
        Records the position of a chain at a certain point in time.
        """
        for var, value in zip(self.varnames, self.f(point)):
            shape = value.shape
            newshape = (1,) + shape
            self.samples[var].append(value.reshape(newshape))
        return self

    def __getitem__(self, index_value):
        """
        Return copy NpTrace with sliced sample values if a slice is passed,
        or the array of samples if a varname is passed.
        """

        if isinstance(index_value, slice):

            sliced_trace = NpTrace(self.vars)
            sliced_trace.samples = dict((name, vals[index_value]) for (name, vals) in self.samples.items())

            return sliced_trace

        else:
            try:
                return self.point(index_value)
            except (ValueError, TypeError, IndexError):
                pass

            return self.samples[str(index_value)]

    def __len__(self):
        rv = self.samples.values()[0].shape[0]
        return rv

    def point(self, index):
        return dict((k, v[index]) for (k, v) in self.samples.items())

    def flush(self):
        self.dbfile.flush()
        return

##########################################################
# MetropolisExt
##########################################################

from pymc.step_methods import Metropolis, NUTS
from pymc.step_methods.arraystep import metrop_select
from pymc.step_methods.metropolis import MultivariateNormalProposal  # tune

from pymc.step_methods.quadpotential import quad_potential
from pymc.step_methods.hmc import HamiltonianMC, bern, energy, leapfrog, Hamiltonian

class MetropolisExt(Metropolis):

    cached_logp = None
    cached_q = None

    def astep(self, q0, logp):
        if self.tune and not self.steps_until_tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = np.atleast_1d(self.proposal_dist() * self.scaling)
        if np.any(self.discrete):
            delta[self.discrete] = round(delta[self.discrete], 0).astype(int)

        q = q0 + delta
        logp_new = logp(q)

        if self.cached_q == None:
            self.cached_q = q0
            self.cached_logp = logp(q0)

        if any(q0 != self.cached_q):
            self.cached_q = q0
            self.cached_logp = logp(q0)
        dif = logp_new - self.cached_logp
        q_new = metrop_select(dif, q, q0)

        if (any(q_new != q0) or all(q0 == q)):
            self.accepted += 1
            self.cached_logp = logp_new
            self.cached_q = q_new

        self.steps_until_tune -= 1
        return q_new

def tune(scale, acc_rate):

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
    '''
    # FIXME
    if scale > 2:
        scale = 2
        print "\nshrink scale\n"
    '''
    return scale
