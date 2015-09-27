#!/usr/bin/env python
import numpy as np
import scipy as sp
import random
import os
import matplotlib.pyplot as plt
import itertools

from srfitext.fitresults import plotResults
from srfitext.utils import saveStruOutput

from srfitext.optimize.bayesian import bayesian
from srfitext.optimize.scipy import scipyLeastsq, scipyFmin, scipyBasinhopping, scipyODR
from srfitext.optimize.de import deapDE, deapGA, RMC


def optimizeExt(recipe, method='leastsq', *args, **kwargs):
    '''input the recipe, optimize, then return the refine result
        rv['x'] is refined parameters
    '''
    # if (plotstep != None) and (plotstep != False):
    #    liveplot = LivePlot(recipe, plotstep, savestru)
    #    plt.ion()
    #    kwargs['callback'] = liveplot

    if method.lower() in ['leastsq']:
        rv = scipyLeastsq(recipe, *args, **kwargs)
    elif method in 'Nelder-Mead Powell CG BFGS Newton-CG Anneal L-BFGS-B TNC COBYLA SLSQP'.split():
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
        rv = bayesian(recipe, *args, **kwargs)

    recipe.residual(rv['x'])

    # if plotstep != None:
    #    plt.ioff()
    return rv


'''class LivePlot(object):

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
            plotResults(
                self.recipe, clf=True, title='No. %d steps' % self.counter)
            if self.savestru != '':
                for contribution in self.recipe._contributions.values():
                    for generator in contribution._generators.values():
                        path = os.path.join(self.savestru, generator.name)
                        saveStruOutput(generator.stru, path)
        return'''