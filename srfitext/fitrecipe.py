#!/usr/bin/env python
import numpy as np
from diffpy.srfit.fitbase import FitRecipe
from srfitext.utils import PConverter
import deap

class FitRecipeExt(FitRecipe):
    '''modified fitrecipe
    residual function and scalarResidual function can be set to adapted max entropy method 
    '''

    def getValues(self):
        """Get the current values of the variables in a list."""
        rv = [v.value for v in self._parameters.values() if
            self.isFree(v)]

        if all([not isinstance(v, np.ndarray) for v in rv]):
            rv = np.array(rv)
        return rv

    def getValuesFlat(self):
        """Get the current values of the variables in a flatten array."""
        rv = [v.value for v in self._parameters.values() if
            self.isFree(v)]
        return self.pconverter.toArray(rv)

    def getBoundsFlat(self):
        lb, ub = self.getBounds2Flat()
        return zip(lb, ub)

    def getBounds2(self):
        """Get the bounds on variables in two lists.

        Returns lower- and upper-bound lists of variable bounds.
        """
        bounds = self.getBounds()
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        if all([not isinstance(v, np.ndarray) for v in lb]):
            lb = np.array(lb)
        if all([not isinstance(v, np.ndarray) for v in ub]):
            ub = np.array(ub)
        return lb, ub

    def getBounds2Flat(self):
        """Get the bounds on variables in two flatten array.

        Returns lower- and upper-bound lists of variable bounds.
        """
        bounds = self.getBounds()
        lb = self.pconverter.toArray([b[0] for b in bounds])
        ub = self.pconverter.toArray([b[1] for b in bounds])
        return lb, ub



    # flag for residual mode, could be 'l2'(default) 'l1'
    _residualmode = 'l2'

    def residual(self, p=[]):
        """Calculate the vector residual to be optimized.

        Arguments
        p   --  The list of current variable values, provided in the same order
                as the '_parameters' list. If p is an empty iterable (default),
                then it is assumed that the parameters have already been
                updated in some other way, and the explicit update within this
                function is skipped.

        The residual is by default the weighted concatenation of each 
        FitContribution's residual, plus the value of each restraint. The array
        returned, denoted chiv, is such that
        
        if self._residualmode == 'l2':
            scalarResidual = dot(chiv, chiv) = chi^2 + restraints (default)
        if self._residualmode == 'l1':
            scalarResidual = sum(chiv) + restraints
        """
        if len(p) != self.pn and (p != []):
            p = self.pconverter.toList(p)

        # Prepare, if necessary
        self._prepare()

        for fithook in self.fithooks:
            fithook.precall(self)

        # Update the variable parameters.
        self._applyValues(p)

        # Update the constraints. These are ordered such that the list only
        # needs to be cycled once.
        for con in self._oconstraints:
            con.update()

        # Calculate the bare chiv
        chiv = np.concatenate([
            np.sqrt(self._weights[i]) * \
                    self._contributions.values()[i].residual().flatten() \
                    for i in range(len(self._contributions))])

        # Calculate the point-average chi^2 or chi depends on self._residualmode
        if self._residualmode == 'l1':
            w = np.average(np.abs(chiv))
        else:  # self._residualmode=='l2':
            w = np.dot(chiv, chiv) / len(chiv)
        # Now we must append the restraints
        penalties = [np.sqrt(res.penalty(w)) for res in self._restraintlist ]
        chiv = np.concatenate([ chiv, penalties ])

        for fithook in self.fithooks:
            fithook.postcall(self, chiv)

        return chiv

    def scalarResidual(self, p=[]):
        """Calculate the scalar residual to be optimized.

        Arguments
        p   --  The list of current variable values, provided in the same order
                as the '_parameters' list. If p is an empty iterable (default),
                then it is assumed that the parameters have already been
                updated in some other way, and the explicit update within this
                function is skipped.

        The residual is by default the weighted concatenation of each 
        FitContribution's residual, plus the value of each restraint. The array
        returned, denoted chiv, is such that 
        
        if self._residualmode == 2: 
            scalarResidual = dot(chiv, chiv) = chi^2 + restraints (default, in leastsq mode)
        if self._residualmode == 1:
            scalarResidual = sum(chi) + restraints (in maxent mode)
        """
        chiv = self.residual(p)
        if self._residualmode == 'l1':
            rv = np.sum(chiv)
        else:  # self._residualmode=='l2':
            rv = np.dot(chiv, chiv)
        return rv

    def scalarResidualTuple(self, p=[]):
        return [self.scalarResidual(p)]

    def setResidualMode(self, mode=None, eq=None):
        '''set how to calculate the residual, also set the equation for each contribution in this recipe
        
        param mode: string, mode of residual, could be 
            leastsq, l2norm: scalarResidual = dot(chiv, chiv) = chi^2 + restraints (default)
            leastabs, l1norm: scalarResidual = sum(chiv) + restraints
        
        param eq: string, equations used for each contribution, could be
            chiv, chi2, x2: (y-y0)/dy -> dot(chiv, chiv) = chi2
            resv, rw2, rw: (y-y0)/sum(y0**2)**0.5 -> dot(resv, resv) = Rw^2
            abs, x1: abs(y-y0)/dy
            logx: abs(log(y/y0))/dy
            xlogx: abs((y-y0)*log(y/y0))/dy
            other equations: directly passed to equation builders in each contributions
            
            *y: calculated profile, y0: observed profile, dy: error of observed profile 
        '''

        if mode != None:
            if mode in ['leastsq', 'l2norm']:
                self._residualmode = 'l2'
            elif mode in ['leastabs', 'l1norm']:
                self._residualmode = 'l1'
        if eq != None:
            if eq in ['chiv', 'chi2', 'x2']:
                eqc = "(eq - _yname)/_dyname"
            elif eq in ['resv', 'rw2', 'rw']:
                eqc = "(eq - _yname)/sum(_yname**2)**0.5"
            elif eq in ['abs', 'x1']:
                eqc = 'abs(eq - _yname)/_dyname'
            elif eq in ['logx']:
                eqc = 'abs(log(eq/_yname))/_dyname'
            elif eq in ['xlogx']:
                eqc = 'abs((eq - _yname) * log(eq/_yname))/_dyname'
            else:
                eqc = eq
            # assign equation to each contributions
            for cont in self._contributions:
                eqstr = eqc.replace('_yname', cont._yname)
                eqstr = eqstr.replace('_dyname', cont._dyname)
                eqstr = eqstr.replace('_xname', cont._xname)
                cont.setResidualEquation(eqstr=eqstr)
        return

    def evaluateODR(self, p=[], x=[]):
        """Calculate the vector value to be optimized, only useful in odr fit.

        need convert self._weights to self._weights**2 before evaluate 
        """
        return self.residual(p)

    def scalarResidualRMC(self, p=[]):
        if len(p) != self.pn and (p != []):
            p = self.pconverter.toList(p)

        # Prepare, if necessary
        self._ready = False
        self._prepare()

        for fithook in self.fithooks:
            fithook.precall(self)

        # Update the variable parameters.
        self._applyValues(p)

        # Update the constraints. These are ordered such that the list only
        # needs to be cycled once.
        for con in self._oconstraints:
            con.update()

        vals = [self._contributions.values()[i].evaluate().flatten() \
                    for i in range(len(self._contributions))]
        yorgs = [self._contributions.values()[i].profile.y.flatten() \
                    for i in range(len(self._contributions))]
        n = vals[0].shape[0]
        chiv = []
        for i in range(len(self._contributions)):
            scale, res, _, _ = np.linalg.lstsq(vals[i].reshape(n, 1), yorgs[i].reshape(n, 1))
            chiv.append(np.sqrt(self._weights[i]) * res[0])

        # Now we must append the restraints
        penalties = [np.sqrt(res.penalty(w)) for res in self._restraintlist ]
        chiv = np.concatenate([ chiv, penalties ])

        for fithook in self.fithooks:
            fithook.postcall(self, chiv)

        return chiv
    
    def gradientScalarResidual(self, p=[]):
        p0 = self.getValuesFlat() if p != [] else p
        derivstep = 0.001
        rv0 = self.scalarResidual(p0)
        var = np.array(p0)
        rv = []
        for k, v in enumerate(var):
            step = derivstep * 0.01 if v == 0 else derivstep * v 
            p0[k] = v + step
            rv1 = self.scalarResidual(p0)
            rv.append((rv1 - rv0) / step)
            p0[k] = v
        return np.r_[rv]

    def fix(self, *args, **kw):
        FitRecipe.fix(self, *args, **kw)
        self.pn = len(self.names)
        self.pconverter = PConverter(self)
        return

    def fixCopy(self, *args, **kw):
        FitRecipe.fix(self, *args, **kw)
        self.pn = len(self.names)
        self.pconverter = PConverter(self, copy=True)
        return

    def free(self, *args, **kw):
        FitRecipe.free(self, *args, **kw)
        self.pn = len(self.names)
        self.pconverter = PConverter(self)
        return

    def freeCopy(self, *args, **kw):
        FitRecipe.free(self, *args, **kw)
        self.pn = len(self.names)
        self.pconverter = PConverter(self, copy=True)
        return

