from pyobjcryst._pyobjcryst import Atom as ObjcrystAtom
from pyobjcryst._pyobjcryst import ScatteringPowerAtom
from pyobjcryst._pyobjcryst import Crystal as ObjcrystCrystal
from pyobjcryst._pyobjcryst import Molecule as ObjcrystMolecule
from diffpy.Structure.atom import Atom as DiffpyAtom
from diffpy.Structure import Structure as DiffpyStructure

from diffpy.srfit.structure.objcrystparset import ObjCrystAtomParSet, ObjCrystMolAtomParSet
from diffpy.srfit.structure.diffpyparset import DiffpyAtomParSet

from srfitext.structure import StructureExt
import deap
import argparse
import numpy as np
import os
import itertools
import re

######################################################################

def getAtomsUsingExp(atoms, exp):
    '''
    exp: string or list of int
    'Mn'->atom.element=='Mn'
    'Mn.face'-> atom.element=='Mn' and atom.face
    'all.x<0.5'-> atom.element== all and atom.x<0.5
    [1,2,3]->atoms[1,2,3]
    '''
    if type(exp) == list:
        rv = [atoms[i] for i in exp]
    elif exp == '':
        rv = []
    elif exp == 'all':
        rv = atoms
    else:
        if len(exp.split('.')) == 1:
            # 'example: Mn or Ti or Ba'
            eles = exp.split(' or ')
            ss = ['atom.element == "%s"' % ele for ele in eles]
            expv = ' or '.join(ss)
            rv = [atom for atom in atoms if eval(expv)]
        else:  # ele.xx, only one element or 'all' is supported
            # get element
            r = re.compile("[a-zA-Z]\w+\.\w+|all")
            allexps = r.findall(exp)
            element = allexps[0].split('.')[0]
            replstr = lambda ss: 'atom.' + ss.split('.')[1] + '.value'
            for rr in allexps:
                expv = exp.replace(rr, replstr(rr))
            if element != 'all':
                expv = '(%s) and (atom.element=="%s")' % (expv, element)
            rv = [atom for atom in atoms if eval(expv)]
    return rv

def resetUiso(atoms):
    for atom in atoms:
        if isinstance(atom, DiffpyAtomParSet):
            if hasattr(atom.atom, 'U'):
                uiso = np.sum(atom.atom.U) / 3
                atom.atom.U = np.eye(3, 3) * uiso
            else:
                aa = atom.atom
                uiso = (aa.B11 + aa.B22, aa.B33) / 3
                aa.B12 = aa.B13 = aa.B23 = 0
                aa.B11 = aa.B22 = aa.B33 = uiso
        elif isinstance(atom, (ObjCrystAtomParSet, ObjCrystMolAtomParSet)):
            sp = atom.scat.GetScatteringPower()
            biso = (sp.B11 + sp.B22 + sp.B33) / 3
            sp.B12 = sp.B13 = sp.B23 = 0
            sp.B11 = sp.B22 = sp.B33 = biso
        else:
            raise TypeError('atom is neither ObjcrystAtomParSet nor DiffpyAtomParSet')
    return

adpUlist = ['Uiso', 'U11', 'U22', 'U33', 'U12', 'U13', 'U23']
adpBlist = ['Biso', 'B11', 'B22', 'B33', 'B12', 'B13', 'B23']

def checkADPUnConstrained(atom, adpname):
    '''check if adp is already constrained
    '''
    rv = True
    if (adpname == 'Biso') or (adpname == 'Uiso'):
        for adp in adpUlist + adpBlist:
            if hasattr(atom, adp):
                if getattr(atom, adp).constrained and rv:
                    rv = False
    elif adpname in adpUlist:
        for adp in ['Uiso', 'Biso', 'B' + adpname[1:]]:
            if hasattr(atom, adp):
                if getattr(atom, adp).constrained and rv:
                    rv = False
    elif adpname in adpBlist:
        for adp in ['Uiso', 'Biso', 'U' + adpname[1:]]:
            if hasattr(atom, adp):
                if getattr(atom, adp).constrained and rv:
                    rv = False
    else:
        raise TypeError('ADP name error')
    return rv

def getElement(obj):
    '''
    get element from diffpy atom or objcryst atom
    '''
    if isinstance(obj, ObjcrystAtom):
        ele = obj.GetScatteringPower().GetSymbol()
    elif isinstance(obj, ScatteringPowerAtom):
        ele = obj.GetSymbol()
    elif isinstance(obj, DiffpyAtom):
        ele = obj.element
    else:
        raise TypeError('atom is neither diffpy atom nor objcryst atom')
    return ele

def saveStruOutput(stru, path):
    path = os.path.splitext(path)[0]
    if isinstance(stru, DiffpyStructure):
        stru.write(path + '.cif', 'cif')
    elif isinstance(stru, ObjcrystCrystal):
        f = file(path + '.cif' , 'w')
        stru.CIFOutput(f)
        f.close()
    elif isinstance(stru, ObjcrystMolecule):
        ss = stru.parent.convertDiffpyStru('xyz_c')
        ss.write(path + '.cif', 'cif')
    elif isinstance(stru, StructureExt):
        stru.convertDiffpyStru('xyz_c').write(path + '.cif', 'cif')

##########
'''process args, including --free, --fix, --method
'''
import argparse
rsparser = argparse.ArgumentParser(description='refine step args parser')
rsparser.add_argument('free', nargs='*', default=[])
rsparser.add_argument('--fix', nargs='*', default=[])
rsparser.add_argument('--method', default='leastsq')
rsparser.add_argument('--plotstep', type=int, default=argparse.SUPPRESS)
rsparser.add_argument('--maxxint', type=int, default=argparse.SUPPRESS)
rsparser.add_argument('--maxiter', type=int, default=argparse.SUPPRESS)

def parseRefineStep(refineargs):
    ''' process refine step args in following format
    (--free) tag1 tag2 tag3 --fix tag4 tag5 --method leastsq
    return a namespace:
    obj.free, obj.fix, obj.method
    
    refineargs could be a string, which will be passed to rsparser and produce an obj
    if it is a list like [str, dict], then the str will be passed to rsparser and dict will
    be assigned to obj.kwargs and finally passed to optimizie function
    '''
    if isinstance(refineargs, str):
        args = refineargs.replace(',', ' ').split()
        rv = rsparser.parse_args(args)
        rv.kwargs = {}
    elif isinstance(refineargs, list):
        args = refineargs[0].replace(',', ' ').split()
        rv = rsparser.parse_args(args)
        rv.kwargs = refineargs[1]
    return rv


############################################################
# converter between list and array
############################################################
def reshapeArray(shape, copy=False):
    def reshapeA(a):
        rv = a.reshape(shape)
        return rv
    def reshapeAc(a):
        rv = a.reshape(shape)
        return np.array(rv)
    return reshapeAc if copy else reshapeA

class PConverter(object):
    '''
    convert p between list and 1D array
    '''

    def __init__(self, recipe, copy=False):
        self.names = recipe.names
        self.values = recipe.values

        self.lens = {}
        self.cdict = {}

        curind = 0
        for n, v in itertools.izip(self.names, self.values):
            if isinstance(v, (float, int)):
                self.cdict[n] = float
                self.lens[n] = [curind, curind + 1]
                curind += 1
            elif isinstance(v, (np.ndarray)):
                ss = map(int, v.shape)
                ni = v.ravel().shape[0]
                self.cdict[n] = reshapeArray(ss, copy=copy)
                self.lens[n] = [curind, curind + ni]
                curind += ni
        return

    def toList(self, parray):
        rv = []
        for n in self.names:
            ll = self.lens[n]
            data = parray[ll[0]:ll[1]]
            d = self.cdict[n](data)
            rv.append(d)

        return rv

    def toArray(self, plist):
        rv = []
        for p in plist:
            if isinstance(p, (float, int)):
                rv.append([p])
            elif isinstance(p, np.ndarray):
                rv.append(p.ravel())
            else:
                raise TypeError('cannot convert list to array')
        rv = np.concatenate(rv)
        return rv
