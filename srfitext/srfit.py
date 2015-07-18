#!/usr/bin/env python
import numpy as np
import itertools
import re, fnmatch, os
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
from srfitext.optimize import optimizeExt

class BaseSrfitExt(object):
    '''class that organize the srfit recipe
    '''

    # some default valuse
    xmin = 1.0
    xmax = 20.0
    dx = 0.01
    parallel = 4
    verbose = 0
    name = 'Base'

    # custom flag
    addname = True
    plotresult = True
    optimizedcalc = False

    '''default bound and default bound scale, when a new variable is created, it
    will first try to find bounds in self.p0, then try self.defaultbounds, then
    try self.defaultboundscale, which will generate bounds by multiple the 
    value of variable by scale value in self.defaultboundscale
    '''
    defaultboundscale = dict({'lat': [0.90, 1.10],
                              'adp': [0.1, 10.0],
                              'xyz': [0.9, 1.1],
                              'scale': [0.1, 10.0],
                              'other': [0.1, 10.0]
                              })
    defaultbounds = dict({'qdamp': [0.06, 0.0, 0.1],
                          'delta2': [5.0, 0.0, 10.0],
                          'delta1': [0.0, 0.0, 5.0],
                          'qbroad': [0.0, 0.0, 0.1],
                          'phaseratio': [1.0, 0.0, 1.0],
                          'occ': [1.0, 0.0, 2.0],
                          })
    # list of pvar, configurations for variables
    pvarlist = []

    # init value
    p0 = dict({'qdamp':  [0.06, 0.0, 0.1],
               'scale':  [0.1, 0.0, 1.0],
               'delta2': [1.0, 0.0, 10.0],
               })
    # restrain
    r0 = dict({
               })
    # refine step
    refinestep = ['scale scalexyz',
                  'adp delta2 occ',
                  'xyz',
                  ]

    def __init__(self, strudict=None, contdict=None, savepath=None, savename=None, **kwargs):
        
        self.savepath = savepath if savepath != None else '.'
        self.savename = savename if savename != None else self.name
        
        self.strudict = strudict
        self.contdict = contdict
        self.recipe = None

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.adpUlist = ['Uiso', 'U11', 'U22', 'U33', 'U12', 'U13', 'U23']
        self.adpBlist = ['Biso', 'B11', 'B22', 'B33', 'B12', 'B13', 'B23']
        self.adplist = self.adpUlist + self.adpBlist
        return

    def loadStru(self, strudict=None):
        '''
        load stru file, the config is provided in strudict
        '''
        strudict = strudict or self.strudict
        self.struext = {}
        self.stru = {}
        strulist = []
        struextlist = []
        
        if isinstance(strudict, dict):
            if strudict.has_key('name'):
                # single stru dict
                stru, struext = self._loadStru(strudict)
                strulist.append(stru)
                struextlist.append(struext)
            else:
                # dict of stru dict, key is the stru name
                for name, sdict in strudict.iteritems():
                    sdict['name'] = name
                    stru, struext = self._loadStru(sdict)
                    strulist.append(stru)
                    struextlist.append(struext)
        elif isinstance(strudict, list):
            # list of stru dict, stru name in the dict
            for sdict in strudict:
                stru, struext = self._loadStru(sdict)
                strulist.append(stru)
                struextlist.append(struext)
        else:
            pass
            # raise ValueError('stru dict not right!!!')
        return strulist, struextlist
    
    def _loadStru(self, strudict):
        '''
        load stru file
        '''
        s = strudict
        
        sfile = s['file']
        name = s.get('name', os.path.splitext(os.path.basename(sfile))[0])
        periodic = s.get('periodic', sfile.endswith('.cif'))
        stype = s.get('type', 'objcryst' if periodic else 'diffpy')
        loadtype = s.get('loadtype', stype if stype in ('diffpy', 'objcryst') else 'diffpy')
        optimized = s.get('optimized', False)
        strukwargs = s.get('strukwargs', {})
        
        struext = StructureExt(name=name, filename=sfile, loadstype=loadtype, periodic=periodic, optimized=optimized, **strukwargs)
        stru = struext.convertStru(stype)
        
        self.stru[name] = stru
        self.struext[name] = struext
        return stru, struext
    
    def GetStru(self, struraw):
        '''
        get the stru list
        '''
        if isinstance(struraw, list):
            if isinstance(struraw[0], str):
                return [self.stru[x] for x in struraw]
            else:
                strulist, struextlist = self.loadStru(struraw)
                return strulist
        else:
            if isinstance(struraw, str):
                return [self.stru[struraw]]
            else:
                strulist, struextlist = self.loadStru(struraw)
                return strulist

    def genContribution(self, contdict):
        '''
        genearte pdf contribution, using contdict
        '''
        contdict = contdict or self.contdict
        self.cont = {}
        rv = []
        
        if isinstance(contdict, dict):
            if contdict.has_key('name'):
                # single cont dict
                cont = self._genCont(contdict)
                rv.append(cont)
            else:
                # dict of cont dict, key is the cont name
                for name, cdict in contdict.iteritems():
                    cdict['name'] = name
                    cont = self._genCont(contdict)
                    rv.append(cont)
        elif isinstance(contdict, list):
            # list of cont dict, cont name in the dict
            for cdict in contdict:
                cont = self._genCont(contdict)
                rv.append(cont)
        
        return rv
    
    def _genCont(self, cdict):
        '''
        generate contribution
        '''
        recipe = self.recipe
        c = cdict
        
        # read meta data from dict
        name = c['name']
        data = c['data']
        weight = c.get('weight', 1.0)
        strulist = self.GetStru(c['stru'])
        xrange = c.get('xrange', (self.xmin, self.xmax, self.dx))
        share = c.get('share', ['qdamp', 'qbroad', 'scale', 'qmin', 'qmax'])  # qmin, qboard, qdamp
        disable = c.get('disable', [])  # do not create var, delta1, delta2 
        
        # make cont
        cont = PDFContributionExt(name, weight)
        self.cont[name] = cont
        cont.setData(data)
        cont.setCalculationRange(xmin=xrange[0], xmax=xrange[1], dx=xrange[2])
        
        # add structures
        for stru in strulist:
            cont.addStructure(stru, parallel=self.parallel)
        for stru in strulist:
            gen = getattr(cont, stru.name)
            self.addVar(gen.delta1, 'delta1_%s' % stru.name, tags=['delta1'])
            self.addVar(gen.delta2, 'delta2_%s' % stru.name, tags=['delta2'])
                
        # assign the scale
        if len(strulist) > 1:
            scalenamelist = ['scale_%s' % stru.name for stru in strulist]
            strunamelist = [stru.name for stru in strulist]
            scalelist = [self.getValue(x)[0] for x in scalenamelist]
            reducedscale = [ss / sum(scalelist) for ss in scalelist]
            for struname, scalename, rscale in itertools.izip(strunamelist[:-1], scalenamelist[:-1], reducedscale[:-1]):
                self.newVar(scalename, [rscale, 0.0, 1.0], tags=['scale', 'phaseratio'])
                recipe.constrain(getattr(cont, struname).scale, 'abs(' + scalename + ') % 1.0')
                # recipe.constrain(getattr(contribution, struname).scale, scalename)
            struname = strulist[-1].name
            recipe.constrain(getattr(cont, struname).scale, '1 - abs(' + ') % 1.0 - abs('.join(scalenamelist[:-1]) + ') % 1.0')
            # recipe.constrain(getattr(contribution, struname).scale, '1 -'+' - '.join(scalenamelist[:-1]))
        
        # add cont to recipe
        recipe.addContribution(cont, cont.weight)
        
        for par in share:
            self.addVar(getattr(cont, par), par)
        for par in ['qdamp', 'qbroad', 'scale', 'qmin', 'qmax']:
            if par not in share:
                self.assignNewVar(getattr(cont, par), par + '_' + cont.name)
        return cont
    
    def makeRecipe(self):
        recipe = FitRecipeExt()
        cdict = self.contdict
        self.recipe = recipe
        recipe.fithooks[0].verbose = self.verbose
        
        if self.strudict:
            self.loadStru(strudict)
        conts = self.genContribution(self.contdict)

        self.makeRecipeAdditional()
        self.processRestrain()
        return recipe

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
            cont = self.recipe._contributions[contribution]
        else:
            cont = contribution
        cont.setData(data)
        return
    
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
        return
    
    def restrainVar(self, varname, restrain=None):
        # [lb, ub, sig, scaled]
        recipe = self.recipe
        rr = restrain if restrain != None else self.r0[varname]
        if len(rr) == 4:
            recipe.restrain(varname, lb=rr[0], ub=rr[1], sig=rr[2], scaled=rr[3])
        elif len(rr) == 3:
            recipe.restrain(varname, lb=rr[0], ub=rr[1], sig=rr[2])
        elif len(rr) == 2:
            recipe.restrain(varname, lb=rr[0], ub=rr[1])
        return
    
    

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
                parname = phasename + '_' + parname if addphasename else parname
                self.addVar(par, parname, par.value, tags=['lat'])
        if xyz:
            for par in sgpars.xyzpars:
                element = getElement(par.par.obj)
                parname = '_'.join([element, par.name])
                if addphasename:
                    parname = '_'.join([phasename, parname])
                self.addVar(par, parname, par.value, tags=['xyz', element + '_xyz'])
        if adp:
            for par in sgpars.adppars:
                element = getElement(par.par.obj)
                parname = '_'.join([element, par.name])
                if addphasename:
                    parname = '_'.join([phasename, parname])
                self.addVar(par, parname, par.value, tags=['adp', element + '_adp'])
        return

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
        recipe = self.recipe
        phase = generator.phase
        atoms = getAtomsUsingExp(phase.getScatterers(), pvar['exps'])
        pvar['par'] = pvar['par'] if type(pvar['par']) == list else [pvar['par']]
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
        pass

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
            elements = map(lambda ele: re.split('[^a-zA-Z]*', ele)[0], elements)
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
        return

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
                        self.assignNewVar(getattr(atom, t), tname, tvalue, tag=['xyz', element + '_xyz'])
                atomn = atomn + 1
        else:
            for atom in atoms:
                for t in ['x', 'y', 'z']:
                    if t in parlist:
                        if pvar.has_key('constrainto'):
                            recipe.constrain(getattr(atom, t), pvar['constrainto'])
                        elif pvar.has_key('var'):
                            recipe.constrain(getattr(atom, t), pvar['var'])
        return

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
            elements = map(lambda ele: re.split('[^a-zA-Z]*', ele)[0], elements)
            varlist = []
            for ele in elements:
                for ux in self.adplist:
                    if ux in parlist:
                        if not '_'.join([ele, ux]) in recipe.names:
                            vv = self.newVar('_'.join([ele, ux]), value, tags=['adp', ele + '_adp'])
                            varlist.append(vv)

        if not (pvar.has_key('var') or pvar.has_key('constrainto')):
            constrainedvarlist = []
            for atom in atoms:
                ele = re.split('[^a-zA-Z]*', atom.element.title())[0]
                for ux in self.adplist:
                    if ux in parlist:
                        if checkADPUnConstrained(atom, ux):
                            recipe.constrain(getattr(atom, ux), 'abs(%s)' % '_'.join([ele, ux]))
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
                            recipe.constrain(getattr(atom, ux), pvar['constrainto'])
                        elif pvar.has_key('var'):
                            recipe.constrain(getattr(atom, ux), pvar['var'])
        return

    # some port of functions
    def assignShapeFunction(self, contribution, generator, shape, appendstr):
        assignShapeFunction(self, contribution, generator, shape, appendstr)
        return

    def assignSASShapeFunction (self, contribution, generator, shape, appendstr):
        assignSASShapeFunction(self, contribution, generator, shape, appendstr)
        return

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
        return


    def setCalculationRange(self, contribution=None, xmin=None, xmax=None, dx=None):
        '''set calculation range of contribution
        if contribution is None, then set range to all contributions
        '''
        xmin = self.xmin if xmin == None else xmin
        xmax = self.xmax if xmax == None else xmax
        dx = self.dx if dx == None else dx
        contribution = self.recipe._contributions if contribution == None else contribution
        contribution = {contribution.name:contribution} if isinstance(contribution, (PDFContributionExt)) else contribution
        for cont in contribution.values():
            cont.setCalculationRange(xmin=xmin, xmax=xmax, dx=dx)
        return
    
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
        return


    def optimize(self, refinestep, **kwargs):
        '''tags are matched using fnmatch
        '''
        recipe = self.recipe
        alltags = self.recipe._tagmanager.alltags()
        if isinstance(refinestep, str):
            rsobj = parseRefineStep(refinestep)
        elif isinstance(refinestep, dict):
            rsobj = parseRefineStep(refinestep['args'])
            kwargs.update(refinestep['kwargs'])
            if kwargs.has_key('filename'):
                if self.savename != None and self.savepath != None \
                        and os.path.split(kwargs['filename'])[0] == '':
                    kwargs['filename'] = os.path.join(self.savepath, self.savename, kwargs['filename'])
                if not os.path.exists(os.path.split(kwargs['filename'])[0]):
                    os.mkdir(os.path.split(kwargs['filename'])[0])

        # free/fix tags
        copy = False
        if rsobj.method in ['ga', 'de']:
            fix = recipe.fixCopy
            free = recipe.freeCopy
        else:
            fix = recipe.fix
            free = recipe.free

        for pattern in rsobj.free:
            matchedtag = fnmatch.filter(alltags, pattern)
            for tag in matchedtag:
                free(tag)
        for pattern in rsobj.fix:
            matchedtag = fnmatch.filter(alltags, pattern)
            for tag in matchedtag:
                fix(tag)
        if hasattr(rsobj, 'maxxint'):
            kwargs['maxxint'] = rsobj.maxxint
        if hasattr(rsobj, 'maxiter'):
            kwargs['maxiter'] = rsobj.maxiter
        rv = optimizeExt(self.recipe, rsobj.method, **kwargs)
        return rv

    def doRefine(self, **kwargs):
        recipe = self.recipe
        # Optimize
        recipe.fix('all')
        for step in self.refinestep:
            rv = self.optimize(step, **kwargs)
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
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        savename = savename if savename != None else self.savename
        recipe = self.recipe
        result = self.result
        filepath = os.path.join(savepath, savename)
        # Save the profiles
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        for contribution in self.recipe._contributions.values():
            profilename = contribution.name
            contribution.profile.savetxt(os.path.join(filepath, profilename + '.fit'))
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
                f = open(filename, 'w')
                f.write('#' + '    '.join(keys))
                np.savetxt(f, gr.transpose(), fmt='%g')
                f.close()
        # Save the structures
        for contribution in self.recipe._contributions.values():
            for generator in contribution._generators.values():
                path = os.path.join(filepath, generator.name)
                saveStruOutput(generator.stru, path)
        # Save the fits results
        result.saveResults(os.path.join(filepath, savename + '.res'))
        # Plot
        # plotResults(recipe, filepath=filepath, title='Rw=%f' % result.rw, show=self.plotresult)
        plotResults(recipe, filepath=filepath, show=self.plotresult)
        # save the bayesian plots if necessary
        try:
            from srfitext.mcmctools import TbTrace
            if isinstance(self.rawresults['raw'], TbTrace):
                trace = self.rawresults['raw']
                os.system('cp %s %s' % (trace.dbfile.filename, os.path.join(filepath, trace.dbfile.filename)))
                if len(trace.varnames) < 25:
                    bayesianPlot(self.rawresults['raw'], filepath=filepath,
                                 show=self.plotresult, shrink=10, burnout=0)
                else:
                    print "there are more than 25 vars, Bayesian plot are disabled"
        except:
            pass
        return result.convals



