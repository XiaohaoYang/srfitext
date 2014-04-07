#!/usr/bin/env python
import numpy as np
import scipy as sp
import random
import os
import matplotlib.pyplot as plt
import itertools

from srfitext.fitresults import plotResults
from srfitext.utils import saveStruOutput

# imports for DE and GA
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.algorithms import varAnd, varOr
from srfitext.gatools import DE, GA
from srfitext.gatools import cxTwoPointsCopy, wirtepop, uniform, mutPolynomialBounded

# imports for pymc
import pymc
from srfitext.mcmctools import MetropolisExt, sample, TbTrace

'''input the recipe, optimize, then return the refine result
    rv['x'] is refined parameters
'''
fminmethodsnames = 'Nelder-Mead Powell CG BFGS Newton-CG Anneal L-BFGS-B TNC COBYLA SLSQP'.split()

class LivePlot(object):

    def __init__(self, recipe, plotstep, savestru=''):
        self.counter = 0
        self.plotstep = plotstep
        self.recipe = recipe
        self.savestru = savestru
        return

    def __call__(self, x, p=None, *args):
        self.counter += 1
        if self.counter % self.plotstep == 0:
            if p != None:
                self.recipe(p)
            plotResults(self.recipe, clf=True, title='No. %d steps' % self.counter)
            if self.savestru != '':
                for contribution in self.recipe._contributions.values():
                    for generator in contribution._generators.values():
                        path = os.path.join(self.savestru, generator.name)
                        saveStruOutput(generator.stru, path)
        return

def optimizeExt(recipe, method='leastsq', plotstep=None, savestru='', *args, **kwargs):
    if (plotstep != None) and (plotstep != False):
        liveplot = LivePlot(recipe, plotstep, savestru)
        plt.ion()
        kwargs['callback'] = liveplot

    if method.lower() in ['leastsq']:
        rv = scipyLeastsq(recipe, *args, **kwargs)
    elif method in fminmethodsnames:
        rv = scipyFmin(recipe, method=method, *args, **kwargs)
    elif method.lower() in ['de']:
        rv = deapDE(recipe, *args, **kwargs)
    elif method.lower() in ['ga']:
        rv = deapGA(recipe, *args, **kwargs)
    elif method.lower() in ['odr']:
        rv = scipyODR(recipe, *args, **kwargs)
    elif method.lower().startswith('bh-'):
        rv = scipyBasinhopping(recipe, method[3:], *args, **kwargs)
    elif method.lower().startswith('bayesian'):
        rv = bayesian(recipe, **kwargs)

    recipe.residual(rv['x'])

    if plotstep != None:
        plt.ioff()
    return rv

##########################################################
# leastsq
##########################################################

def scipyLeastsq(recipe, *args, **kwargs):
    from scipy.optimize import leastsq
    print "Fit using scipy's LM optimizer"
    leastsq_kwargs = {}
    if kwargs.has_key('maxiter'):
        leastsq_kwargs['maxfev'] = kwargs['maxiter']
    if kwargs.has_key('callback'):
        print "Live plotting not supported in Scipy leastsq"
    # p, pcov, infodict, errmsg, ier = leastsq(recipe.residual, recipe.getValues(), full_output=1, factor=10, **leastsq_kwargs)
    p, pcov, infodict, errmsg, ier = leastsq(recipe.residual, recipe.getValuesFlat(),
                                             full_output=1, factor=10, **leastsq_kwargs)
    return {'x':p,
            'cov':pcov}

##########################################################
# Fmin
##########################################################

def scipyFmin(recipe, method='BFGS', *args, **kwargs):
    '''method available:
        Nelder-Mead
        Powell
        CG
        BFGS
        Newton-CG
        Anneal
        L-BFGS-B
        TNC
        COBYLA
        SLSQP
    bounds only available for  L-BFGS-B, TNC, COBYLA and SLSQP
    '''
    from scipy.optimize import minimize
    print 'Fit using scipy fmin %s optimizer' % method
    fmin_kwargs = {}
    if kwargs.has_key('maxiter'):
        fmin_kwargs['options'] = {'maxiter':kwargs['maxiter'], }
    if kwargs.has_key('callback'):
        fmin_kwargs['callback'] = kwargs['callback']
    # res = minimize(recipe.scalarResidual, recipe.getValues(), method=method, bounds=recipe.getBounds(), *args, **fmin_kwargs)
    res = minimize(recipe.scalarResidual, recipe.getValuesFlat(),
                   method=method, bounds=recipe.getBoundsFlat(), *args, **fmin_kwargs)
    return {'x': res['x'],
            'raw': res}

##########################################################
# Basinhopping
##########################################################

class MyBounds(object):
    def __init__(self, recipe):
        # self.xmin, self.xmax = recipe.getBounds2()
        self.xmin, self.xmax = recipe.getBounds2Flat()
        return

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

class MyRandomDisplacement(object):
    def __init__(self, recipe, stepsize=0.5):
        self.recipe = recipe
        # self.xmin, self.xmax = recipe.getBounds2()
        self.xmin, self.xmax = recipe.getBounds2Flat()
        self.range = self.xmax - self.xmin
        self.stepsize = stepsize
        return

    def __call__(self, x):
        res0 = self.recipe.scalarResidual(x)
        dx = np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)) * self.range
        res = self.recipe.scalarResidual(x + dx)
        i = 0
        while (res > (res0 * 2)) and (i < 20):
            dx = np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)) * self.range
            res = self.recipe.scalarResidual(x + dx)
            i += 1
        x += dx
        return x

def scipyBasinhopping(recipe, method='L-BFGS-B', *args, **kwargs):
    # new in scipy 0.12
    from scipy.optimize import basinhopping
    print "Fit using scipy's basin hopping optimizer"
    mybounds = MyBounds(recipe)
    mystep = MyRandomDisplacement(recipe)
    minimizer_kwargs = {'method': method,
                        # 'bounds': recipe.getBounds(),
                        'bounds': recipe.getBoundsFlat(),
                        'options': {'maxiter':300},
                        }
    if kwargs.has_key('maxiter'):
        minimizer_kwargs['options'] = kwargs['maxiter']

    bh_kwargs = {'take_step': mystep,
                 'accept_test': mybounds}
    if kwargs.has_key('callback'):
        bh_kwargs['callback'] = kwargs['callback']
    if kwargs.has_key('maxxint'):
        bh_kwargs['niter'] = kwargs['maxxint']
    else:
        bh_kwargs['niter'] = 20

    res = basinhopping(recipe.scalarResidual, recipe.getValues(),
                       minimizer_kwargs=minimizer_kwargs,
                       **bh_kwargs)
    return {'x': res['x'],
            'raw': res}

##########################################################
# ODR
##########################################################

def scipyODR(recipe, *args, **kwargs):
    from scipy.odr import Data, Model, ODR, RealData, odr_stop

    # FIXME
    # temporarily change _weights to _weights**2 to fit the ODR fits
    a = [w ** 2 for w in recipe._weights]
    recipe._weights = a

    model = Model(recipe.evaluateODR,
                  # implicit=1,
                  meta=dict(name='ODR fit'),
                  )
    x = [recipe._contributions.values()[0].profile.x]
    y = [recipe._contributions.values()[0].profile.y]
    dy = [recipe._contributions.values()[0].profile.dy]

    cont = recipe._contributions.values()
    for i in range(1, len(cont)):
        xplus = x[-1][-1] - x[-1][-2] + x[-1][-1]
        x.append(cont[i].profile.x + x[-1][-1] + xplus)
        y.append(cont[i].profile.y)
        dy.append(cont[i].profile.dy)
    x.append(np.arange(len(recipe._restraintlist)) * (x[-1][-1] - x[-1][-2]) + x[-1][-1])
    y.append(np.zeros_like(recipe._restraintlist))
    dy = np.concatenate(dy)
    dy = np.concatenate([dy, np.ones_like(recipe._restraintlist) + np.average(dy)])
    data = RealData(x=np.concatenate(x), y=np.concatenate(y), sy=dy)

    odr_kwargs = {}
    if kwargs.has_key('maxiter'):
        odr_kwargs['maxit'] = kwargs['maxiter']
    odr = ODR(data, model, beta0=recipe.getValues(), **odr_kwargs)
    odr.set_job(deriv=1)
    out = odr.run()
    # out.pprint()

    # FIXME
    # revert back
    a = [np.sqrt(w) for w in recipe._weights]
    recipe._weights = a

    return {'x': out.beta,
            'esd': out.sd_beta,
            'cov': out.cov_beta,
            'raw': out,
            }

##########################################################
# Differential evolution
##########################################################

def preprocess(recipe):
    recipe.verbose = 0

    creator.create("FitnessM", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessM)
    # init toolbox()
    toolbox = base.Toolbox()
    bl, bh = recipe.getBounds2Flat()
    toolbox.register("attr_float", uniform, bl, bh)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # extra
    toolbox.bl = bl
    toolbox.bh = bh
    return toolbox

def deapDE(recipe, cr=0.25, f=1, mu=300, ngen=20, *args, **kwargs):
    '''differential evolution using deap package
    '''
    ndim = len(recipe.getValues())
    toolbox = preprocess(recipe)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", recipe.scalarResidualTuple)

    rv = DE(toolbox, ndim, cr, f, mu, ngen, *args, **kwargs)
    wirtepop('pop.dat', rv['hof'])
    return rv

def deapGA(recipe, cxpb=0.5, mutpb=0.2, ngen=40, mu=2000, *args, **kwargs):
    '''differential evolution using deap package
    '''
    recipe.verbose = 0

    toolbox = preprocess(recipe)
    toolbox.register("mate", cxTwoPointsCopy)
    toolbox.register("mutate", mutPolynomialBounded, eta=1.0, low=toolbox.bl, up=toolbox.bh, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", recipe.scalarResidualTuple)

    rv = GA(toolbox, cxpb, mutpb, ngen, mu, *args, **kwargs)
    wirtepop('pop.dat', rv['hof'])
    return rv

##########################################################
# Differential evolution
##########################################################

def RMC(recipe, steps=50000, *args, **kwargs):

    liveplot = LivePlot(recipe, plotstep=100, savestru='')
    cont = recipe._contributions.values()[0]
    gen = cont._generators.values()[0]
    phase = gen._phase
    gen._calc.evaluatortype = 'OPTIMIZED'

    plt.ion()

    counter = 0
    rescurr = recipe.scalarResidualRMC()[0]
    while counter < steps:
        phase.xyz.notify()
        counter += 1
        # recipe.scale = counter % 10.0
        randi = np.random.randint(-1, phase.n)
        xyz = phase.xyz.value[randi]
        newxyz = xyz + np.random.normal(0, 1, 3) * 0.1
        phase.updateXYZi(randi, newxyz)
        resnew = recipe.scalarResidualRMC()[0]
        if resnew >= rescurr:
            deltares = resnew - rescurr
            rand = np.random.exponential() * 0.01
            if deltares > rand:
                phase.updateXYZi(randi, newxyz)
                # print "%f rejected" % resnew
            else:
                print "%f accepted" % resnew
                rescurr = resnew
        else:
            print "%f accepted" % resnew
            rescurr = resnew
        # liveplot(0)
    saveStruOutput(gen.stru, 'fitresult')
    return

##########################################################
# Bayesian
##########################################################

def bayesian(recipe, steps=50000, S=None, *args, **kwargs):
    # mcfunc: evaluate
    # dmcfunc: evaluate differential

    names = recipe.names
    values = recipe.values
    lb, ub = recipe.getBounds2()
    d = {}
    for i in range(len(values)):
        d1 = values[i] - lb[i]
        d2 = ub[i] - values[i]
        d[names[i]] = np.minimum(d1, d2)

    with pymc.Model() as model:
        for i in range(len(names)):
            if isinstance(values[i], np.ndarray):
                shape = values[i].shape
                pymc.Uniform(names[i], lower=lb[i], upper=ub[i], shape=shape)
            else:
                pymc.Uniform(names[i], lower=lb[i], upper=ub[i])

        start = dict(zip(names, values))
        # determine the scale for proposal distribution
        if S == None:
            S = np.concatenate([[d[v.name] / 2] if isinstance(d[v.name], float) else d[v.name].ravel() / 2 for v in model.vars])
        step = MetropolisExt(model.vars, S=S, tune_interval=10000)
        # step.scaling = 0.5
        logp = step.fs[0]
        def logpMod(f):
            p = [float(f[n]) if f[n].shape == () else f[n] for n in names]
            return -recipe.scalarResidual(p) + logp(f)
        # dlogp = step.fs[1]
        # def dlogpMod(f):
        #    return dmcfunc(f) + dlogp(f)

        # step.ordering.vmap
        step.fs = [logpMod]
        # step.fs = [logpMod, dlogpMod]

    # run sampler
    with model:
        trace = TbTrace(model.unobserved_RVs, **kwargs)
        trace = sample(steps, step, start, trace=trace)

    # take mean as the output
    rv = []
    for n in names:
        rv.append(np.mean(trace[n], axis=0))
    rv = recipe.pconverter.toArray(rv)

    return {'x': rv,
            'raw': trace
            }


