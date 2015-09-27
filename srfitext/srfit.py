#!/usr/bin/env python
import numpy as np
import itertools
import re
import fnmatch
import os
import sys
import matplotlib.pyplot as plt

from diffpy.srfit.fitbase import FitContribution, FitRecipe, FitResults, Profile
from diffpy.srfit.structure.diffpyparset import DiffpyStructureParSet
from diffpy.srfit.structure.objcrystparset import ObjCrystCrystalParSet, ObjCrystAtomParSet

from srfitext.structure import StructureExt
from srfitext.contribution import PDFContributionExt
from srfitext.fitrecipe import FitRecipeExt
from srfitext.fitresults import FitResultsExt, plotResults, bayesianPlot
from srfitext.generator import PDFGeneratorExt
from srfitext.struparset import StructureExtParSet, ObjCrystMoleculeParSetExt

from srfitext.shapefunction import assignShapeFunction, assignSASShapeFunction
from srfitext.utils import parseRefineStep, resetUiso, getAtomsUsingExp, \
    checkADPUnConstrained, getElement, saveStruOutput
from srfitext.optimizemethod import optimizeExt


class BaseSrfitExt(object):
    '''class that organize the srfit recipe
    '''
    parallel = 4
    verbose = 0

    xrange = (1.0, 20.0, 0.01)
    plotresult = True
    optimizedcalc = False

    '''default bound and default bound scale, when a new variable is created, it
    will first try to find bounds in self.p0, then try self.defaultbounds, then
    try self.defaultboundscale, which will generate bounds by multiple the 
    value of variable by scale value in self.defaultboundscale
    '''
    defaultboundscale = {'lat': [0.90, 1.10],
                         'adp': [0.1, 10.0],
                         'xyz': [0.9, 1.1],
                         'scale': [0.1, 10.0],
                         'other': [0.1, 10.0]}
    defaultbounds = {'qdamp': [0.06, 0.0, 0.1],
                     'delta2': [5.0, 0.0, 10.0],
                     'delta1': [0.0, 0.0, 5.0],
                     'qbroad': [0.0, 0.0, 0.1],
                     'phaseratio': [1.0, 0.0, 1.0],
                     'occ': [1.0, 0.0, 2.0]}
    # list of pvar, configurations for variables
    pvarlist = []

    # init value
    p0 = {'qdamp': [0.06, 0.0, 0.1],
          'scale': [0.1, 0.0, 1.0],
          'delta2': [1.0, 0.0, 10.0]}
    # restrain
    r0 = {}
    # refine step
    refinestep = ['scale scalexyz',
                  'adp delta2 occ',
                  'xyz', ]

    def __init__(self, struConfig=None, contributionConfig=None, savepath=None, savename=None, **kwargs):

        self.savepath = savepath if savepath != None else '.'
        self.savename = savename if savename != None else 'refine'

        self.struConfig = struConfig
        self.contributionConfig = contributionConfig
        self.recipe = None

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.adpUlist = ['Uiso', 'U11', 'U22', 'U33', 'U12', 'U13', 'U23']
        self.adpBlist = ['Biso', 'B11', 'B22', 'B33', 'B12', 'B13', 'B23']
        self.adplist = self.adpUlist + self.adpBlist

    def loadStru(self, struConfig):
        '''
        load stru file, the config is provided in struConfig
        '''
        self.struext = {}
        self.stru = {}
        rv = []

        if isinstance(struConfig, dict):
            return [self._loadStru(struConfig)]
        elif isinstance(struConfig, (list, tuple)):
            for sdict in struConfig:
                rv.append(self._loadStru(sdict))
            return rv
        else:
            raise ValueError('stru dict not right!!!')

    def _loadStru(self, struConfig):
        '''
        load stru file
        '''
        s = struConfig

        sfile = s['file']
        name = s.get('name', os.path.splitext(os.path.basename(sfile))[0])
        periodic = s.get('periodic', sfile.endswith('.cif'))
        stype = s.get('type', 'objcryst' if periodic else 'diffpy')
        loadtype = s.get(
            'loadtype', stype if stype in ('diffpy', 'objcryst') else 'diffpy')
        optimized = s.get('optimized', False)
        strukwargs = s.get('strukwargs', {})

        self.struext[name] = StructureExt(
            name=name, filename=sfile, loadstype=loadtype, periodic=periodic, optimized=optimized, **strukwargs)
        self.stru[name] = self.struext[name].convertStru(stype)
        return name

    def genContribution(self, cConfig):
        '''
        genearte pdf contribution, using contributionConfig
        '''
        self.contribution = {}

        if isinstance(cConfig, dict):
            self._genContribution(cConfig)
        elif isinstance(cConfig, (list, tuple)):
            for cd in cConfig:
                self.cConfig(cc)

    def _genContribution(self, cConfig):
        '''
        generate contribution
        '''
        recipe = self.recipe
        # read meta data from dict
        name = cConfig['name']
        data = cConfig['data']
        weight = cConfig.get('weight', 1.0)
        pdfxrange = cConfig.get('xrange', self.xrange)
        # qmin, qboard, qdamp
        sharedVar = cConfig.get(
            'sharedVar', ['qdamp', 'qbroad', 'scale', 'qmin', 'qmax'])
        # do not create var, delta1, delta2
        disabledVar = cConfig.get('disabledVar', [])

        # load stru
        struConfig = cConfig.get('stru')
        if struConfig is not None:
            struNameList = self.loadStru(struConfig)
            strulist = [self.stru[n] for n in self.stru]
        else:
            strulist = self.stru.items()

        # make contribution
        contribution = PDFContributionExt(name=name, weight=weight)
        self.contribution[name] = contribution
        contribution.setData(data)
        contribution.setCalculationRange(
            xmin=pdfxrange[0], xmax=pdfxrange[1], dx=pdfxrange[2])

        # add structures
        for stru in strulist:
            contribution.addStructure(stru, parallel=self.parallel)
        for stru in strulist:
            gen = getattr(contribution, stru.name)
            self.addVar(gen.delta1, '%s_delta1' % stru.name, tags=['delta1'])
            self.addVar(gen.delta2, '%s_delta2' % stru.name, tags=['delta2'])

        # assign the scale
        if len(strulist) > 1:
            scalenamelist = ['scale_%s' % stru.name for stru in strulist]
            strunamelist = [stru.name for stru in strulist]
            scalelist = [self.getValue(x)[0] for x in scalenamelist]
            reducedscale = [ss / sum(scalelist) for ss in scalelist]
            for struname, scalename, rscale in itertools.izip(strunamelist[:-1], scalenamelist[:-1], reducedscale[:-1]):
                self.newVar(
                    scalename, [rscale, 0.0, 1.0], tags=['scale', 'phaseratio'])
                recipe.constrain(
                    getattr(contribution, struname).scale, 'abs(' + scalename + ') % 1.0')
            struname = strulist[-1].name
            recipe.constrain(getattr(
                contribution, struname).scale, '1 - abs(' + ') % 1.0 - abs('.join(scalenamelist[:-1]) + ') % 1.0')

        # add cont to recipe
        recipe.addContribution(contribution, contribution.weight)

        for par in sharedVar:
            self.addVar(getattr(contribution, par), par)
        for par in ['qdamp', 'qbroad', 'scale', 'qmin', 'qmax']:
            if par not in sharedVar:
                self.assignNewVar(
                    getattr(cont, par), par + '_' + contribution.name)
        return name

    def makeRecipe(self):
        self.recipe = FitRecipeExt()
        self.recipe.fithooks[0].verbose = self.verbose

        if self.struConfig:
            self.loadStru(struConfig)
        self.genContribution(self.contributionConfig)

        self.makeRecipeAdditional()
        self.processRestrain()

    def makeRecipeAdditional(self):
        '''add some additional var, constrain to recipe, called in self.makerecipe
        '''
        return

    def setData(self, data, contribution):
        '''set the profile of contribution

        param contribution: NPPDFContribution, or str, 
            if NPPDFContribution, data will be assigned to this contributions 
            if string, which is the name of contribution, data will assigned to
            self.recipe._contributions[contribution]
        param data: string or list of array
            if string, the profile will be read from this file
            if list of array or (2D array), then the first row profile[0] will
            be x, the second row profile[1] will be y and third row profile[2]
            (if exists) will be dy
        '''
        if isinstance(contribution, str):
            c = self.recipe._contributions[contribution]
        else:
            c = contribution
        c.setData(data)

    def getValue(self, varname, varvalue=None, tag='all'):
        '''get value and boundary of var

        return [value, lowbound, highbound]
        '''
        # get value from p0
        if self.p0.has_key(varname):
            return self.p0[varname]
        else:
            if varvalue is None:
                if self.defaultbounds.has_key(tag):
                    return self.defaultbounds[tag]
                else:
                    raise ValueError('Define var in p0 or provide varvalue')
            else:
                if isinstance(varvalue, (list, tuple)):
                    return varvalue
                else:
                    if self.defaultbounds.has_key(tag):
                        vl, vh = self.defaultbounds[tag]
                    elif self.defaultboundscale.has_key(tag):
                        vl, vh = self.defaultboundscale[tag]
                        vl = vl * varvalue
                        vh = vh * varvalue
                    else:
                        vl = varvalue * 0.8
                        vh = varvalue * 1.2
                    return (varvalue, vl, vh)

    def newVar(self, varname, varvalue=None, tags=['all']):
        '''
        add new var and constrain to var
        '''
        v, vl, vh = self.getValue(varname, varvalue, tags[0])
        rv = self.recipe.newVar(varname, v, tags=tags)
        rv.bounds = [vl, vh]
        return rv

    def addVar(self, par, varname, varvalue=None, tags=['all']):
        '''
        add new var to self.recipe
        '''
        v, vl, vh = self.getValue(varname, varvalue, tags[0])
        rv = self.recipe.addVar(par, v, varname, tags=tags)
        rv.bounds = [vl, vh]
        return rv

    def processRestrain(self):
        for varname in self.r0.keys():
            self.restrainVar(varname)

    def restrainVar(self, varname, restrain=None):
        # [lb, ub, sig, scaled]
        recipe = self.recipe
        rr = restrain if restrain != None else self.r0[varname]
        if len(rr) == 4:
            recipe.restrain(
                varname, lb=rr[0], ub=rr[1], sig=rr[2], scaled=rr[3])
        elif len(rr) == 3:
            recipe.restrain(varname, lb=rr[0], ub=rr[1], sig=rr[2])
        elif len(rr) == 2:
            recipe.restrain(varname, lb=rr[0], ub=rr[1])

    def returnGenerator(self, gen):
        '''
        return the generator from the name
        '''
        if isinstance(gen, PDFGeneratorExt):
            return gen
        elif isinstance(gen, str):
            for cont in self.recipe._contributions.itervalues():
                if cont._generators.has_key(gen):
                    return cont._generators[gen]

    def returnContribution(self, cont):
        '''
        return the contribution
        '''
        if isinstance(cont, PDFContributionExt):
            return cont
        elif isinstance(cont, str):
            if self.recipe._contributions.has_key(cont):
                return self.recipe._contributions[cont]

    def constrainAsSpaceGroup(self, generator, spacegroup, lat=True, xyz=False, adp=False, addphasename=False):
        '''
        constrain phase using space group

        param generator: generator.phase in recipe to constrain
        param spacegroup: space group in number or string "87" or "F m 3 m"
        param lat: bool, constrain lattice parameters
        param xyz: bool, constrain xyz of atoms
        param adp: bool, constrain adp of atoms
        param addphasename: bool, if add phase name to name of variable
        '''

        from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup, _constrainAsSpaceGroup
        from diffpy.Structure.spacegroupmod import SpaceGroup

        generator = self.returnGenerator(generator)
        recipe = self.recipe
        phase = generator.phase
        phasename = generator.name

        if isinstance(phase, DiffpyStructureParSet):
            if isinstance(spacegroup, SpaceGroup):
                sgpars = _constrainAsSpaceGroup(phase, spacegroup)
            else:
                sgpars = constrainAsSpaceGroup(phase, spacegroup)
        elif isinstance(phase, ObjCrystCrystalParSet):
            sgpars = phase.sgpars

        if lat:
            for par in sgpars.latpars:
                parname = par.name
                parname = phasename + '_' + \
                    parname if addphasename else parname
                self.addVar(par, parname, par.value, tags=['lat'])
        if xyz:
            for par in sgpars.xyzpars:
                element = getElement(par.par.obj)
                parname = '_'.join([element, par.name])
                parname = phasename + '_' + \
                    parname if addphasename else parname
                self.addVar(
                    par, parname, par.value, tags=['xyz', element + '_xyz'])
        if adp:
            for par in sgpars.adppars:
                element = getElement(par.par.obj)
                parname = '_'.join([element, par.name])
                parname = phasename + '_' + \
                    parname if addphasename else parname
                self.addVar(
                    par, parname, par.value, tags=['adp', element + '_adp'])

    def processVar(self, generator, pvar):
        '''assign var to all atoms match pvar
        pvar: dict with keys:
            'exps': expressions to filter atoms, could be 'all' or '*' for all atoms
            'par': name of parameter to be constrained, could be
                adp(biso), biso, b11, b22, b33, occ, x, y, z, xyz(x, y, z)
            'var': basename of variable to be refined, actual name would be element+var
                if None, would be element+par
            'value': value of var
            'bound': bound of var
            'constrainto': name of variable or expression that constrained to, 
                example: use ['B11', 'B22'] in 'var' and 'B11' here will constrain 
                atom.B11 and atom B22 to variable B11

            examples:
            {'exps':'all', 'par':['Biso', 'occ']}: wil assign uiso and occ to all atoms
                assign differnet type of parameters only avaiable if 'var' 'constrainto' is None
            {'exps':'Mn', 'par':['B11', 'B22'], 'var':'Mn_u11'}: constrain Mn.U11 and Mn.U22 with
                variable 'Mn_u11'
            # FIXME {'exps':{'z': ['=',0],'element':['=','Mn']}, 'par':['y'], 'constrainto':'Mn_x'}:
                constrain Mn.y of all Mn atoms with Mn.z==0 to 'Mn_x' (assuming 'Mn_x' has been 
                created
            {'exps':'Ti', 'par':['occ'], 'var':'Ti_occ_1'}
                constrain Ti.occ to Ti_occ_1
            {'exps':'Mn', 'par:['occ'], 'constrainto':'1-Ti_occ_1'}:
                constrain Mn.occ to 1-Ti_occ_1 (assuming Ti_occ_1 has been created
         }
        '''
        generator = self.returnGenerator(generator)
        recipe = self.recipe
        phase = generator.phase
        atoms = getAtomsUsingExp(phase.getScatterers(), pvar['exps'])
        pvar['par'] = pvar['par'] if type(
            pvar['par']) == list else [pvar['par']]
        parlist = set(pvar['par'])
        if 'occ' in parlist:
            self._assignOCC(atoms, pvar)
        # diffpy structure
        if any([ux in parlist for ux in self.adplist]):
            if ('Biso' in parlist) or ('Uiso' in parlist):
                resetUiso(atoms)
            self._assignADP(atoms, pvar)
        if any([xx in parlist for xx in ['x', 'y', 'z']]):
            self._assignXYZ(atoms, pvar)

    def _assignOCC(self, atoms, pvar):
        '''assign occ to atom in atoms
        '''
        recipe = self.recipe
        if pvar.has_key('var'):
            if pvar['var'].lower() != 'none':
                value = pvar['value'] if pvar.has_key('value') else None
                self.newVar(pvar['var'], value, tags=['occ'])
        else:
            elements = list(set([atom.element for atom in atoms]))
            elements = map(
                lambda ele: re.split('[^a-zA-Z]*', ele)[0], elements)
            for ele in elements:
                if ele + '_occ' not in recipe.names:
                    self.newVar(ele + '_occ', tags=['occ'])
        for atom in atoms:
            if pvar.has_key('constrainto'):
                recipe.constrain(atom.occ, pvar['constrainto'])
            elif pvar.has_key('var'):
                recipe.constrain(atom.occ, pvar['var'])
            else:
                element = re.split('[^a-zA-Z]*', atom.element.title())[0]
                recipe.constrain(atom.occ, element + '_occ')

    def _assignXYZ(self, atoms, pvar):
        '''assign xyz to atom in atoms
        '''
        parlist = [pp.lower() for pp in pvar['par']]
        recipe = self.recipe
        if pvar.has_key('var'):
            if pvar['var'].lower() != 'none':
                value = pvar['value'] if pvar.has_key('value') else None
                bound = pvar['bound'] if pvar.has_key('bound') else None
                self.addNewVar(pvar['var'], value, bound, tag='xyz')
        atomn = 0
        if not (pvar.has_key('var') or pvar.has_key('constrainto')):
            for atom in atoms:
                element = re.split('[^a-zA-Z]*', atom.element.title())[0]
                for t in ['x', 'y', 'z']:
                    if t in parlist:
                        tvalue = getattr(atom, t).value
                        tname = '_'.join([element, t, str(atomn)])
                        self.assignNewVar(
                            getattr(atom, t), tname, tvalue, tag=['xyz', element + '_xyz'])
                atomn = atomn + 1
        else:
            for atom in atoms:
                for t in ['x', 'y', 'z']:
                    if t in parlist:
                        if pvar.has_key('constrainto'):
                            recipe.constrain(
                                getattr(atom, t), pvar['constrainto'])
                        elif pvar.has_key('var'):
                            recipe.constrain(getattr(atom, t), pvar['var'])

    def _assignADP(self, atoms, pvar):
        '''assign adp to atom in atoms
        Biso and B11,B22,B33, first come first constrained
        '''
        recipe = self.recipe
        parlist = pvar['par']
        value = pvar['value'] if pvar.has_key('value') else None
        if pvar.has_key('var'):
            if pvar['var'].lower() != 'none':
                self.newVar(pvar['var'], value, tags=['adp'])
        else:
            elements = list(set([atom.element for atom in atoms]))
            elements = map(
                lambda ele: re.split('[^a-zA-Z]*', ele)[0], elements)
            varlist = []
            for ele in elements:
                for ux in self.adplist:
                    if ux in parlist:
                        if not '_'.join([ele, ux]) in recipe.names:
                            vv = self.newVar(
                                '_'.join([ele, ux]), value, tags=['adp', ele + '_adp'])
                            varlist.append(vv)

        if not (pvar.has_key('var') or pvar.has_key('constrainto')):
            constrainedvarlist = []
            for atom in atoms:
                ele = re.split('[^a-zA-Z]*', atom.element.title())[0]
                for ux in self.adplist:
                    if ux in parlist:
                        if checkADPUnConstrained(atom, ux):
                            recipe.constrain(
                                getattr(atom, ux), 'abs(%s)' % '_'.join([ele, ux]))
                            constrainedvarlist.append('_'.join([ele, ux]))
            # delete unconstrained variables
            constrainedvarlist = set(constrainedvarlist)
            for varr in varlist:
                if not varr.name in constrainedvarlist:
                    recipe.delVar(varr)
        else:
            for atom in atoms:
                for ux in self.adplist:
                    if ux in parlist:
                        if pvar.has_key('constrainto'):
                            recipe.constrain(
                                getattr(atom, ux), pvar['constrainto'])
                        elif pvar.has_key('var'):
                            recipe.constrain(getattr(atom, ux), pvar['var'])

    # some port of functions
    def assignShapeFunction(self, contribution, generator, shape, appendstr=''):
        assignShapeFunction(self, contribution, generator, shape, appendstr)

    def assignSASShapeFunction(self, contribution, generator, shape, appendstr=''):
        assignSASShapeFunction(self, contribution, generator, shape, appendstr)

    # zoom scale of structure
    def assignZoomscale(self, generator):
        '''
        assign a zoom scale to non-perodic structure
        '''
        recipe = self.recipe
        phase = generator._phase
        zname = 'zoomscale_' + generator.name
        self.newVar(zname, varvalue=[1.0, 0.0, 2.0], tags=['scale'])
        if isinstance(phase, DiffpyStructureParSet):
            lattice = generator._phase.getLattice()
            recipe.constrain(lattice.a, zname)
            recipe.constrain(lattice.b, zname)
            recipe.constrain(lattice.c, zname)
        elif isinstance(phase, (StructureExtParSet, ObjCrystMoleculeParSetExt)):
            recipe.constrain(phase.zoomscale, zname)
        else:
            raise TypeError('Cannot assign zoomscale!')

    def setCalculationRange(self, contribution, xmin, xmax, dx):
        '''set calculation range of contribution
        if contribution is None, then set range to all contributions
        '''
        if isinstance(contribution, list):
            cont = [self.returnContribution(x) for x in contribution]
        else:
            cont = [self.returnContribution(contribution)]
        for c in cont:
            c.setCalculationRange(xmin=xmin, xmax=xmax, dx=dx)

    # optimize
    def setResidualMode(self, recipe=None, model=None, eq=None):
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
        recipe = self.recipe if recipe == None else recipe
        recipe.setResidualMode(model, eq)

    def optimize(self, refinestep, **kwargs):
        '''tags are matched using fnmatch
        '''
        recipe = self.recipe
        alltags = self.recipe._tagmanager.alltags()

        freetags = refinestep.get('free', '')
        freetags = re.split(r'[;,\s]\s*', freetags) if freetags else []
        fixtags = refinestep.get('fix', '')
        fixtags = re.split(r'[;,\s]\s*', fixtags) if fixtags else []

        for pattern in freetags:
            for tag in fnmatch.filter(alltags, pattern):
                recipe.free(tag)
        for pattern in fixtags:
            for tag in fnmatch.filter(alltags, pattern):
                recipe.fix(tag)
        kwargs.update(refinestep)
        # method is in refinestep
        rv = optimizeExt(self.recipe, **kwargs)
        return rv

    def doRefine(self):
        recipe = self.recipe
        # Optimize
        recipe.fix('all')
        for step in self.refinestep:
            rv = self.optimize(step)
        try:
            self.result = FitResultsExt(recipe, raw=rv)
            self.result.printResults()
            rv['rw'] = self.result.rw
        except:
            pass
        self.rawresults = rv
        return rv

    def saveResult(self, savepath=None, savename=None):
        savepath = savepath if savepath != None else self.savepath
        savename = savename if savename != None else self.savename
        filepath = os.path.join(savepath, savename)
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        recipe = self.recipe
        result = self.result

        # Save the profiles
        for contribution in self.recipe._contributions.values():
            profilename = contribution.name
            contribution.profile.savetxt(
                os.path.join(filepath, profilename + '.fit'))
            # save gr of different generators
            if len(contribution._generators.values()) > 1:
                r = contribution.profile.x
                grs = [r, contribution.profile.y]
                keys = ['r', 'total']
                for generator in contribution._generators.values():
                    grs.append(generator(r) * contribution.scale.value)
                    keys.append(generator.name)
                gr = np.vstack(grs)
                filename = os.path.join(filepath, profilename + '.sep.fit')
                with open(filename, 'w') as f:
                    f.write('#' + '    '.join(keys))
                    np.savetxt(f, gr.transpose(), fmt='%g')

        # Save the structures
        for contribution in self.recipe._contributions.values():
            for generator in contribution._generators.values():
                path = os.path.join(filepath, generator.name)
                saveStruOutput(generator.stru, path)
        # Save the fits results
        result.saveResults(os.path.join(filepath, savename + '.res'))

    def plotResult(self, savepath=None, savename=None):
        savepath = savepath if savepath != None else self.savepath
        savename = savename if savename != None else self.savename
        filepath = os.path.join(savepath, savename)
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        recipe = self.recipe
        result = self.result

        plotResults(recipe, filepath=filepath, show=self.plotresult)
        # save the bayesian plots if necessary
        try:
            '''from .optimizemethod.pymcbackend import HDF5file
            if isinstance(self.rawresults['raw'], TbTrace):
                trace = self.rawresults['raw']
                os.system('cp %s %s' % (trace.dbfile.filename, os.path.join(filepath, trace.dbfile.filename)))'''
            # if len(trace.varnames) < 25:
            bayesianPlot(self.rawresults['raw'], filepath=filepath,
                         show=self.plotresult, shrink=10, burnout=0)
            # else:
            # print "there are more than 25 vars, Bayesian plot are disabled"
        except:
            pass
