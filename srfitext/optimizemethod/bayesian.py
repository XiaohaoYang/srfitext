import numpy as np
import pymc3
from .pymcbackend import HDF5file
from .pymcsample import sample, iter_sample 
from .pymcstepmethod import MetropolisExt
# from pymc3.step_methods import Metropolis
import itertools

def bayesian(recipe, steps=20000, tune_interval=2000, stepmethod='metropolis', logpscale=0.01, distributions={}, *args, **kwargs):
    # mcfunc: evaluate
    # dmcfunc: evaluate gradient

    names = recipe.names
    values = recipe.values
    lbs, ubs = recipe.getBounds2()
    
    dist = {}
    for name, value, lb, ub in zip(names, values, lbs, ubs):
        temp = np.minimum(value - lb, ub - value)
        sd = temp / 2 if isinstance(temp, float) else temp.ravel() / 2
                
        if not distributions.has_key(name):
            kw = {'dist':'uniform',
                  'lower':lb,
                  'upper':ub,
                  'transform':None}
        else:
            kw = distributions[name]
            if kw['dist'] == 'uniform':
                kw.setdefault('lower', lb)
                kw.setdefault('upper', ub)
                kw.setdefault('transform', None)
            elif kw['dist'] == 'normal':
                sd = kw.get('sd', sd)
                kw.setdefault('mu', value)
                kw.setdefault('sd', sd)
            else:
                raise ValueError('dist type not supported')
        dist[name] = kw

    with pymc3.Model() as model:
        for name in names:
            kw = dist[name].copy()
            d = kw.pop('dist')
            if d == 'uniform':
                pymcdist = pymc3.Uniform
                print 'Uniform distribution assigned for %s' % name
            elif d == 'normal':
                pymcdist = pymc3.Normal
                print 'Normal distribution assigned for %s' % name
            else:
                raise ValueError("distribution type error, currently only support uniform or normal")
            pymcdist(name, **kw)

        start = dict(zip(names, values))
            
        if stepmethod == 'metropolis':
            step = MetropolisExt(model.vars, tune_interval=tune_interval, recipe=recipe, logpscale=logpscale)
        else:
            raise ValueError('stepmethod not support')
    
    # run sampler
    with model:
        trace = HDF5file(name='bayesian', model=model, vars=None, filename='mcmc.h5', loadfromfile=False)
        trace = sample(steps, step, start, trace=trace)

    # take mean as the output
    rv = []
    for n in names:
        rv.append(np.mean(trace[n], axis=0))
    rv = np.array(rv)
    # rv = recipe.pconverter.toArray(rv)

    return {'x': rv,
            'raw': trace
            }
