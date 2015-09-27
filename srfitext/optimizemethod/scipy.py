#!/usr/bin/env python
##########################################################
# LeastSq
##########################################################


def scipyLeastsq(recipe, *args, **kwargs):
    from scipy.optimize import leastsq
    print "Fit using scipy LM optimizer"
    leastsq_kwargs = {}
    if kwargs.has_key('maxiter'):
        leastsq_kwargs['maxfev'] = kwargs['maxiter']
    if kwargs.has_key('callback'):
        print "Live plotting not supported in Scipy leastsq"
    p, pcov, infodict, errmsg, ier = leastsq(recipe.residual, recipe.values,
                                             full_output=1, factor=10, **leastsq_kwargs)
    return {'x': p,
            'cov': pcov}

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
        fmin_kwargs['options'] = {'maxiter': kwargs['maxiter'], }
    if kwargs.has_key('callback'):
        fmin_kwargs['callback'] = kwargs['callback']
    res = minimize(recipe.scalarResidual,
                   recipe.values,
                   method=method,
                   bounds=recipe.getBoundsFlat(),
                   *args, **fmin_kwargs)
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
        dx = np.random.uniform(-self.stepsize,
                               self.stepsize, np.shape(x)) * self.range
        res = self.recipe.scalarResidual(x + dx)
        i = 0
        while (res > (res0 * 2)) and (i < 20):
            dx = np.random.uniform(-self.stepsize,
                                   self.stepsize, np.shape(x)) * self.range
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
                        'options': {'maxiter': 300},
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
    x.append(np.arange(len(recipe._restraintlist))
             * (x[-1][-1] - x[-1][-2]) + x[-1][-1])
    y.append(np.zeros_like(recipe._restraintlist))
    dy = np.concatenate(dy)
    dy = np.concatenate(
        [dy, np.ones_like(recipe._restraintlist) + np.average(dy)])
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
