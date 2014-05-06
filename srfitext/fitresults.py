#!/usr/bin/env python
import numpy as np
import numpy
import os
from diffpy.srfit.fitbase.fitresults import FitResults, ContributionResults, ContributionResults
import matplotlib.pyplot as plt

class FitResultsExt(FitResults):
    '''change the uncertainty estimation in original FitResults
    multiply rchi2 to get estimation when data unceratinty is unavailable
    '''
    def __init__(self, recipe, update=True, showfixed=True, showcon=
            False, raw=None):
        """Initialize the attributes.

        recipe   --  The recipe containing the results
        update  --  Flag indicating whether to do an immediate update (default
                    True).
        showcon --  Show fixed variables in the output (default True).
        showcon --  Show constraint values in the output (default False).
        
        raw: raw results output from refinements
        """
        recipe.scalarResidual(np.array(raw['x']))
        super(FitResultsExt, self).__init__(recipe, update, showfixed, showcon)
        self.raw = raw
        return


    def update(self):
        """Update the results according to the current state of the recipe."""
        self.n = len(self.recipe.getNames())
        self.hasarray = any([isinstance(x, np.ndarray) for x in self.recipe.getValues()])
        
        if not self.hasarray:
            FitResults.update(self)
            # correct cov/std if all dy == 1
            certain = True
            for con in self.conresults.values():
                if (con.dy == 1).all():
                    certain = False
            if not certain:
                self.cov = self.cov * self.rchi2
                sqrtrchi2 = np.sqrt(self.rchi2)
                self.varunc = [old * sqrtrchi2 for old in self.varunc]
                self.conunc = [old * sqrtrchi2 for old in self.conunc]
                for con in self.conresults.values():
                    con.conunc = [old * sqrtrchi2 for old in con.conunc]
            
            # convert array to NAN
            self.varvals = [np.NAN if isinstance(v, np.ndarray) else v for v in self.varvals]
            self.fixedvals = [np.NAN if isinstance(v, np.ndarray) else v for v in self.fixedvals]
        else:
            self.recipe._prepare()
            res = self.recipe.residual()
            self.residual = numpy.dot(res, res)
            for con, weight in zip(self.recipe._contributions.values(), self.recipe._weights):
                self.conresults[con.name] = ContributionResultsExt(con, weight, self)
            self._calculateMetrics()
        return


    def saveResults(self, filename, header="", footer="", update=False):
        """Format and save the results.

        filename -  Name of the save file.
        header  --  A header to add to the output (default "")
        footer  --  A footer to add to the output (default "")
        update  --  Flag indicating whether to call update() (default False).

        """
        # Save the time and user
        from time import ctime
        from getpass import getuser
        myheader = "Results written: " + ctime() + "\n"
        myheader += "produced by " + getuser() + "\n"
        header = myheader + header

        res = self.formatResults(header, footer, update)
        f = file(filename, 'w')
        f.write(res)
        if self.raw != None:
            f.write(str(self.raw))
        f.close()
        return
    
class ContributionResultsExt(ContributionResults):
    
    def _init(self, con, weight, fitres):
        """Initialize the attributes, for real."""
        # # Note that the order of these operations is chosen to reduce
        # # computation time.
        if con.profile is None:
            return
        recipe = fitres.recipe
        # Store the weight
        self.weight = weight
        # First the residual
        res = con.residual()
        self.residual = numpy.dot(res, res)
        # The arrays
        self.x = numpy.array(con.profile.x)
        self.y = numpy.array(con.profile.y)
        self.dy = numpy.array(con.profile.dy)
        self.ycalc = numpy.array(con.profile.ycalc)
        # The other metrics
        self._calculateMetrics()
        # Find the parameters
        if len(recipe._oconstraints) == len(fitres.convals):
            for i, constraint in enumerate(recipe._oconstraints):
                par = constraint.par
                loc = con._locateManagedObject(par)
                if loc:
                    self.conlocs.append(loc)
                    self.convals.append(fitres.convals[i])
                    self.conunc.append(fitres.conunc[i])

        return

def plotResults(recipe, filepath=None, clf=True, title='plot', show=False):
    """Plot the results contained within a refined FitRecipe."""

    # All this should be pretty familiar by now.
    names = recipe.getNames()
    vals = recipe.getValues()
    i = 1
    for contribution in recipe._contributions.values():
        r = contribution.profile.x
        g = contribution.profile.y
        gcalc = contribution.profile.ycalc
        diffzero = -0.8 * max(g) * np.ones_like(g)
        diff = g - gcalc + diffzero

        plt.figure(i)
        if clf:
            plt.clf()
        i = i + 1
        plt.plot(r, g, 'b-', lw=3, label="G(r) Data")
        plt.plot(r, gcalc, 'r-', lw=2, label="G(r) Fit")
        plt.plot(r, diff, 'g-', lw=2, label="G(r) diff")
        plt.plot(r, diffzero, 'k-')
        plt.xlabel("$r (\AA)$")
        plt.ylabel("$G (\AA^{-2})$")
        plt.title(title)
        plt.legend(loc=1)
        if filepath != None:
            plt.savefig(os.path.join(filepath, contribution.name + '_fits.png'))
            plt.savefig(os.path.join(filepath, contribution.name + '_fits.pdf'))
    if show:
        plt.show()
    return

def bayesianPlot(trace, filepath, show=False, shrink=1, burnout=0):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    varnames = trace.varnames
    for varn in varnames:
        data = trace.samples[varn][burnout::shrink]
        plt.figure(varn)
        plt.clf()
        plt.title(varn)
        plt.hist(data, bins=30, normed=True)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(filepath, varn + '.png'))
    if show:
        plt.show()
    pass
