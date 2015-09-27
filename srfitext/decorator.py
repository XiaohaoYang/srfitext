#!/usr/bin/env python
from nputils import getAtomsUsingExp
import numpy as np
import scipy.spatial.distance as ssd
import itertools
# from easyfit.npfit.npatom import NPAtom, NPAtomTet
# from easyfit.npfit.npstructure import NPStructure

class BaseDecorator(object):
    '''Structure decorator
    when init, it will register to one stru instance, and link the variable in 
    self.param to stru instance
    so careful about namespace conflict
    
    param name: str, name of decorator
    param params: dict, parameters register to the stru instance and then register to generator,
        these parameters are refinable in the refinement
    param paramext: dict, parameters used in decorator with their default value,
        these parameters are not refinable 
    '''
    name = 'decorator'
    _params = {}
    _extparams = {}

    def __init__(self, name=None):
        if name != None:
            self.name = name
        for param in self._params.keys():
            setattr(self, param, self._params[param])
        for param in self._extparams.keys():
            setattr(self, param, self._extparams[param])
        return

    def decorate(self, stru):
        '''process stru and return it 
        Overload this method
        '''
        return stru

    def __call__(self, stru):
        return self.decorate(stru)

##################################################
# some decorators
##################################################


##########################################################################################




class CdSeDecorator(BaseDecorator):
    '''scale xyz of all atoms by scalexyz
    '''
    box1 = np.array([0.25, 0.0, 0.5])
    box2 = np.array([0.75, 0.5, 1.0])
    def decorate(self, stru):
        ss = []
        for atom in stru:
            if all(atom.xyz < self.box2) and all(atom.xyz > self.box1):
                pass
            else:
               ss.append(atom)
        for atom in ss:
            stru.remove(atom)
        return stru

########################################################
class LayerDecorator(BaseDecorator):
    '''set layer structure
    '''
    name = 'layer'
    _params = {
            'direction': 'c',
            'distance': 100,
            }
    _extparams = {
            'number': 1,
            'zoom':1,
            }

    def decorate(self, stru):
        a = stru.lattice.a
        b = stru.lattice.b
        c = stru.lattice.c

        if self.direction == 'a':
            ijklist = [(a, 0, 0) for a in range(self.number)]
            mnofloats = np.array((self.number + self.distance / a, 1, 1), dtype=float)
        elif self.direction == 'b':
            ijklist = [(0, b, 0) for b in range(self.number)]
            mnofloats = np.array((1, self.number + self.distance / b, 1), dtype=float)
        elif self.direction == 'c':
            ijklist = [(0, 0, c) for c in range(self.number)]
            mnofloats = np.array((1, 1, self.number + self.distance / c), dtype=float)

        # build a list of new atoms
        newAtoms = []
        for ijk in ijklist:
            for a in S:
                adup = NPAtom(a)
                adup.xyz = (a.xyz + ijk) / mnofloats
                newAtoms.append(adup)
        # newS can own references in newAtoms, no need to make copies
        stru.__setslice__(0, len(newS), newAtoms, copy=False)

        # take care of lattice parameters
        stru.lattice.setLatPar(
                a=mnofloats[0] * a,
                b=mnofloats[1] * b,
                c=mnofloats[2] * c)

        return stru

########################################################
class RemoveLabeledAtomDecorator(BaseDecorator):
    '''remove atoms with label in self.removelabeledatom
    '''
    name = 'removelabeledatom'
    _params = {'removelabeledatom': []}

    def decorate(self, stru):
        atoms = [atom for atom in stru.tolist() if self.matchLabel(atom, self.removelabeledatom)]
        for atom in atoms:
            stru.remove(atom)
        return stru

    def matchLabel(self, atom, labels):
        ll = [label for label in labels if atom.label.startswith(label)]
        rv = True if ll else False
        return rv

########################################################
class RemoveAtomDecorator(BaseDecorator):
    '''remove atoms according to their attributes
    Example: ['Cd_c'] will remove 'Cd' atom whose corner is True
    '''
    name = 'removeatom'
    _params = {'removeatom':''}

    def decorate(self, stru):
        atoms = getAtomsUsingExp(stru.tolist(), self.removeatom)
        for atom in set(atoms):
            stru.remove(atom)
        return stru


########################################################
class RandomLigandDecorator(BaseDecorator):
    '''attach ligand random to the surface of NP.
    class attributes
        ligandbond: float, bond length of ligand, can be refined
        ligandnumber: int, number of ligands
        ligandsites: list of str, or list of expressions, possible sites of ligands
        
    '''
    name = 'randomligand'
    _params = {
            'ligandbond':1.43,
            'ligandnumber':0,
            }
    _extparams = {
            'ligand':NPStructure(),
            'nearestbondrange':[0, 1.5],
            'ligandbond_sigma':0.1,
            'ligandsites': 'Cd',
            'cn': 4,
            }

    def decorate(self, stru):
        '''update xyz of ligand atoms
        '''
        for xyz, abg in itertools.izip(self.sitelist, self.abglist):
            rotmatrix = self.rotationMatrix(abg[0], abg[1], abg[2], mode='ZYZ')
            rxyz = np.dot(rotmatrix, self.ligandxyz).T
            newligand = self.ligand.__copy__()
            newligand.xyz_cartn = xyz + rxyz
            stru.extend(newligand)
        return stru

    def initSite(self, metadata=None, mode=None):
        metadata = 'ligand.npz' if metadata == None else metadata
        if mode == 'load':
            obj = np.load(metadata)
            self.sitelist = obj['sitelist']
            self.abglist = obj['abglist']
        elif mode == 'save':
            self._initSite(metadata, True)
        else:
            self._initSite()
        self.ligandxyz = np.array(self.ligand.xyz_cartn.T)
        return

    def _initSite(self, metadata=None, save=False):
        '''generate list of atom sites for ligand to attach on, then randomly choose a direction 
        for each ligand.
        '''
        stru = self.stru
        center = np.mean(stru.xyz_cartn, axis=0)
        cxyz = stru.xyz_cartn

        sitelist = []
        for atom in self.stru:
            xyz = np.array(atom.xyz_cartn)
            distlist = np.array([ssd.euclidean(xyz, nxyz) for nxyz in cxyz])
            ind = np.logical_and(distlist > self.nearestbondrange[0],
                                 distlist < self.nearestbondrange[1])
            number = np.sum(ind)
            if (number < self.cn) and (atom.element == self.ligandsites):
                sitelist.append(xyz)

        sitelist = np.array(sitelist)
        sitelist = np.random.permutation(sitelist)[:self.ligandnumber]

        abglist = []
        for i in range(len(sitelist)):
            xyz = sitelist[i]
            dirvec = xyz - center
            dist = ssd.euclidean(xyz, center)
            theta = np.arccos(dirvec[2] / dist)
            phi = np.arctan2(dirvec[1], dirvec[0])
            ll = dist + np.random.normal(self.ligandbond, self.ligandbond_sigma)
            sitelist[i] = center + dirvec / dist * ll
            abglist.append([phi, theta, 0])

        self.sitelist = sitelist
        abglist = np.array(abglist)
        if abglist.shape[0] > 0:
            abglist[:, 2] = np.random.rand(abglist.shape[0]) * np.pi * 2
        self.abglist = abglist

        if save:
            metadata = 'ligand.npz' if metadata == None else metadata
            np.savez(metadata, sitelist=sitelist, abglist=abglist)

        return

    def rotationMatrix(self, alpha, beta, gamma, mode='ZYZ'):
        s = np.sin(np.array([alpha, beta, gamma]))
        c = np.cos(np.array([alpha, beta, gamma]))
        if mode == 'ZXZ':
            rv = [[c[0] * c[2] - c[1] * s[0] * s[2], -c[0] * s[2] - c[1] * c[2] * s[0], s[0] * s[1]],
                  [c[2] * s[0] + c[0] * c[1] * s[2], c[0] * c[1] * c[2] - s[0] * s[2], -c[0] * s[1]],
                  [s[1] * s[2], c[2] * s[1], c[1]      ]]
        elif mode == 'ZYZ':
            rv = [[c[0] * c[1] * c[2] - s[0] * s[2], -c[2] * s[0] - c[0] * c[1] * s[2], c[0] * s[1]],
                  [c[0] * s[2] + c[1] * c[2] * s[0], c[0] * c[2] - c[1] * s[0] * s[2], s[0] * s[1]],
                  [-c[2] * s[1], s[1] * s[2], c[1]      ]]
        elif mode == 'rot':
            # rotate about rho&theta->alpha,beta?
            rv = [[c[1] * c[0], c[2] * s[0] + s[2] * s[1] * c[0], s[2] * s[0] - c[2] * s[1] * c[0]],
                  [-c[1] * s[0], c[2] * c[0] + s[2] * s[1] * s[0], s[2] * c[0] - c[2] * s[1] * s[0]],
                  [s[1], -s[2] * c[1], c[2] * c[1]]]
        return np.array(rv)

class RandomLigandDecorator_alt(RandomLigandDecorator):
    '''attach ligand random to the surface of NP.
    '''
    name = 'randomligand'
    _params = {
            'ligand1bond': 2.26,
            'ligand2bond': 2.35,
            'ligand1number':0,
            'ligand2number':0,
            }
    _extparams = {
            'ligand1':NPStructure(),
            'ligand2':NPStructure(),
            'nearestbondrange':[0, 3.0],
            'ligandbond_sigma':0.001,
            'ligandsites': 'Cd',
            'cn': 4,
            }

    def rotApend(self, stru, ligand, rotm, xyz):
        rxyz = np.dot(rotm, ligand.xyz_cartn.T).T
        newligand = ligand.__copy__()
        newligand.xyz_cartn = xyz + rxyz
        stru.extend(newligand)
        return

    def calDirvec(self, xyz, stru):
        '''calcualte the direction of ligand, return phi and theta
        '''
        Selist = [atom for atom in stru if atom.element == 'Se']
        Sexyzlist = [np.array(atom.xyz_cartn) for atom in Selist]
        distlist = np.array([ssd.euclidean(xyz, atom.xyz_cartn) for atom in Selist])
        ind = np.logical_and(distlist > self.nearestbondrange[0],
                             distlist < self.nearestbondrange[1])
        nearest_xyz = np.array(Sexyzlist)[ind]
        dirvec = np.average(nearest_xyz, axis=0)
        dirvec = xyz - dirvec

        theta = np.arccos(dirvec[2] / np.sqrt(np.sum(dirvec ** 2)))
        phi = np.arctan2(dirvec[1], dirvec[0])
        dirvec = dirvec / np.sqrt(np.sum(dirvec ** 2))
        return phi, theta, dirvec


    def decorate(self, stru):
        '''update xyz of ligand atoms
        '''
        center = np.mean(stru.xyz_cartn, axis=0)
        cxyz = stru.xyz_cartn

        sitelist1 = np.random.permutation(len(self.sitelist))[:self.ligand1number]
        sitelist2 = np.random.permutation(len(self.sitelist))[:self.ligand2number]
        doublesite0 = set(sitelist1) & set(sitelist2)
        singlesite1 = [self.sitelist[i] for i in (set(sitelist1) - doublesite0)]
        singlesite2 = [self.sitelist[i] for i in (set(sitelist2) - doublesite0)]
        doublesite = [self.sitelist[i] for i in doublesite0]

        for xyz in singlesite1:
            phi, theta, dirvec = self.calDirvec(xyz, stru)
            bl = np.random.normal(self.ligand1bond, self.ligandbond_sigma)
            ligandsite = xyz + bl * dirvec

            rotm1 = self.rotationMatrix(0, np.random.normal(0.1, 0.1), 0)
            rotm2 = self.rotationMatrix(phi, theta, np.random.rand() * np.pi * 2)
            rotm = np.dot(rotm2, rotm1)
            self.rotApend(stru, self.ligand1, rotm, ligandsite)

        for xyz in singlesite2:
            phi, theta, dirvec = self.calDirvec(xyz, stru)
            bl = np.random.normal(self.ligand2bond, self.ligandbond_sigma)
            ligandsite = xyz + bl * dirvec

            rotm = self.rotationMatrix(phi, theta, np.random.rand() * np.pi * 2)
            self.rotApend(stru, self.ligand2, rotm, ligandsite)

        for xyz in doublesite:
            # fake site
            phi, theta, dirvec = self.calDirvec(xyz, stru)

            deg1 = np.radians(30)
            deg2 = np.radians(-30)
            dirvecO = np.array([np.sin(deg1), 0, np.cos(deg1)]) * np.random.normal(self.ligand1bond, self.ligandbond_sigma)
            dirvecN = np.array([np.sin(deg2), 0, np.cos(deg2)]) * np.random.normal(self.ligand2bond, self.ligandbond_sigma)
            rot = self.rotationMatrix(phi, theta, np.random.rand() * np.pi * 2)
            dirvecO = np.dot(rot, dirvecO.T).T
            dirvecN = np.dot(rot, dirvecN.T).T

            siteO = xyz + dirvecO
            siteN = xyz + dirvecN

            theta = np.arccos(dirvecO[2] / np.sqrt(np.sum(dirvecO ** 2)))
            phi = np.arctan2(dirvecO[1], dirvecO[0])
            rotm1 = self.rotationMatrix(0, np.random.normal(0.1, 0.1), 0)
            rot1 = self.rotationMatrix(phi, theta, np.random.rand() * np.pi * 2)
            rotm = np.dot(rot1, rotm1)
            self.rotApend(stru, self.ligand1, rotm, siteO)

            theta = np.arccos(dirvecN[2] / np.sqrt(np.sum(dirvecN ** 2)))
            phi = np.arctan2(dirvecN[1], dirvecN[0])
            rotm = self.rotationMatrix(phi, theta, np.random.rand() * np.pi * 2)
            self.rotApend(stru, self.ligand2, rotm, siteN)

        return stru

    def initSite(self):
        '''generate list of atom sites for ligand to attach on
        '''
        stru = self.stru
        sitelist = []
        for atom in self.stru:
            xyz = np.array(atom.xyz_cartn)
            distlist = np.array([ssd.euclidean(xyz, nxyz) for nxyz in stru.xyz_cartn])
            ind = np.logical_and(distlist > self.nearestbondrange[0],
                                 distlist < self.nearestbondrange[1])
            number = np.sum(ind)
            if (number < self.cn) and (atom.element == self.ligandsites):
                sitelist.append(xyz)

        self.sitelist = sitelist
        self.ligand1.xyz_cartn = self.ligand1.xyz_cartn + np.array([1.082530, 0, 0.625])
        return

