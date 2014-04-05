'''assign shape function to recipe
'''
from diffpy.srfit.pdf.characteristicfunctions import SASCF

import numpy as np
import itertools
import re


def bnn(r, psize, ratio):
    f = np.zeros(np.shape(r), dtype=float)
    if psize > 0:
        x = np.array(r, dtype=float) / psize
        inside = (x < 1.0)
        xin = x[inside]
        f[inside] = 1.0 - 1.5 * xin + 0.5 * xin * xin * xin
        f *= ratio
        f += 1
    return f

def sheetCF(r, sthick, ratio):
    """Nanosheet characteristic function.
    
    r       --  distance of interaction
    sthick  --  Thickness of nanosheet
    
    From Kodama et al., Acta Cryst. A, 62, 444-453

    """
    if sthick <= 0: return np.zeros_like(r)

    f = 0.5 * sthick / r
    sel = (r <= sthick)
    f[sel] = 1 - (r / sthick / 2)[sel]
    f *= ratio
    f += 1
    return f

# default parameters used in _registerFunc
_shapeFunctionMeta = \
            {'sphere':
                {'name':'sphericalCF',
                 'psize':[20, 0, 100], },
            'spheroid':
                {'name':'spheroidalCF',
                 'erad':[10, 0, 100],
                 'prad':[10, 0, 100], },
            'spheroid2':
                {'name':'spheroidalCF2',
                 'psize':[20, 0, 100],
                 'axrat':[1.0, 0, 10.0], },
            'logshpere':
                {'name':'lognormalSphericalCF',
                 'psize':[20, 0, 100],
                 'psig':[1, 0, 10], },
            'sheet':
                {'name':'sheetCF',
                 'func': sheetCF,
                 'sthick':[20, 0, 100],
                 'ratio': [0.5, 0, 1.0]},
            'shell':
                {'name':'shellCF',
                 'radius':[10, 0, 100],
                 'thickness':[10, 0, 100], },
            'bnn':
                {'name':'bnn',
                 'func': bnn,
                 'psize': [20, 0, 100],
                 'ratio': [0.5, 0, 1.0]},
            }


def assignShapeFunction(recipe, contribution, generator, shape, appendstr=''):
    ''' sphere:
            psize   --  The particle diameter
        spheroid:
            prad    --  polar radius
            erad    --  equatorial radius
        spheroid2:
            psize  --  The equatorial diameter
            axrat  --  The ratio of axis lengths
        logshpere:
            psize  --  The mean particle diameter
            psig   --  The log-normal width of the particle diameter
        sheet:
            sthick  --  Thickness of nanosheet
        shell:
            radius      --  Inner radius
            thickness   --  Thickness of shell
    '''
    import diffpy.srfit.pdf.characteristicfunctions
    funcdata = _shapeFunctionMeta[shape]
    funcname = funcdata['name']
    if funcdata.has_key('func'):
        func = funcdata['func']
    else:
        func = getattr(diffpy.srfit.pdf.characteristicfunctions, funcname)

    contribution.registerFunction(func, name=funcname)

    equationstr = contribution.eqstr
    generatorname = generator.name
    equationstr = equationstr.replace(' ' + generatorname + ' ', ' ( %s * %s ) ' % (funcname, generatorname))
    contribution.eqstr = equationstr
    contribution.setEquation(equationstr)
    varlist = func.func_code.co_varnames[1:func.func_code.co_argcount]
    for varname in varlist:
        varobj = getattr(contribution, varname)
        recipe.assignNewVar(varobj, varname + appendstr,
                            varvalue=funcdata[varname],
                            tag='shape', p0first=True)
    return

_sasShapeFunctionMeta = \
        {'Sphere':
            {'name':'SphereModel',
             'radius':[20, 0, 100], },
        'Ellipsoid':
            {'name':'EllipsoidModel',
             'radius_a':[10, 0, 100],
             'radius_b':[10, 0, 100], },
        'TriaxialEllipsoid':
            {'name':'TriaxialEllipsoidModel',
             'semi_axisA':[20.0, 0, 50.0],
             'semi_axisB':[30.0, 0, 50.0],
             'semi_axisC':[40.0, 0, 50.0], },
        'FuzzySphere':
            {'name':'FuzzySphereModel',
             'radius':[30, 0, 100],
             'fuzziness':[10, 0, 50], },
        'Hardsphere':
            {'name':'HardsphereStructure',
             'effect_radius':[30, 0, 100],
             'volfraction':[0.2, 0, 1.0], },
        'Cylinder':
            {'name':'CylinderModel',
             'radius':[20, 0, 100],
             'length':[20, 0, 100], },
        'EllipticalCylinder':
            {'name':'EllipticalCylinderModel',
             'r_minor':[20, 0, 100],
             'r_ratio':[1.5, 0, 4],
             'length':[100, 0, 400], },
        'HollowCylinder':
            {'name':'HollowCylinderModel',
             'core_radius':[20.0, 0, 100.0],
             'radius':[30.0, 0, 20.0],
             'length':[100, 0, 400], },
        'BCCrystal':
            {'name':'BCCrystalModel',
             'dnn':[10, 0, 100],
             'd_factor':[10, 0, 100],
             'radius':[20, 0, 100], },
        'FCCrystal':
            {'name':'FCCrystalModel',
             'dnn':[10, 0, 100],
             'd_factor':[10, 0, 100],
             'radius':[20, 0, 100], },
        'SSCrystal':
            {'name':'SCCrystalModel',
             'dnn':[10, 0, 100],
             'd_factor':[10, 0, 100],
             'radius':[20, 0, 100], },
        'LamellarFFHG':
            {'name':'LamellarFFHGModel',
             't_length':[15.0, 0, 50],
             'h_thickness':[10, 0, 50], },
        'Lamellar':
            {'name':'LamellarModel',
             'bi_thick':[30.0, 0, 100.0], },
        'LamellarPCrystal':
            {'name':'LamellarPCrystal',
             'thickness':[20.0, 0, 100.0],
             'Nlayers':[20, 0, 100.0],
             'spacing':[100, 0, 400],
             'pd_spacing':[0.0, 0, 1.0], },
        'LamellarPS':
            {'name':'LamellarPSModel',
             'delta':[20.0, 0, 50.0],
             'spacing':[50.0, 0, 400.0],
             'caille':[0.1, 0, 1.0], },
        'SquareWell':
            {'name':'SquareWellStructure',
             'effect_radius':[20.0, 0, 50.0],
             'volfraction':[0.04, 0.0, 1.0],
             'welldepth':[1.5, 0.0, 3.0],
             'wellwidth':[1.5, 0.0, 3.0], },
        'BarBell':
            {'name':'BarBellModel',
             'rad_bar':[10, 0, 100],
             'len_bar':[100, 0, 400],
             'rad_bell':[20, 0, 100], },
        'BinaryHS':
            {'name':'BinaryHSModel',
             'l_radius':[100, 0, 400],
             's_radius':[20, 0, 100], },
        'BinaryHSPSF11':
            {'name':'BinaryHSPSF11Model',
             'l_radius':[100, 0, 400],
             's_radius':[20, 0, 100], },
        'CappedCylinder':
            {'name':'CappedCylinderModel',
             'rad_cyl':[20, 0, 100],
             'len_cyl':[100, 0, 400],
             'rad_cap':[20, 0, 100], },
        'DiamCyl':
            {'name':'DiamCylFunc',
             'radius':[20, 0, 100],
             'length':[100, 0, 400], },
        'DiamEllip':
            {'name':'DiamEllipFunc',
             'radius_a':[20, 0, 100],
             'radius_b':[100, 0, 400], },
        'FlexCylEllipX':
            {'name':'FlexCylEllipXModel',
             'length':[100, 0, 400],
             'kuhn_length':[50, 0, 100],
             'radius':[20, 0, 100],
             'axis_ratio':[1.5, 0, 4], },
        'FlexibleCylinder':
            {'name':'FlexibleCylinderModel',
             'length':[100, 0, 400],
             'kuhn_length':[50, 0, 100],
             'radius':[20, 0, 100], },
        'Fractal':
            {'name':'FractalModel',
             'radius':[5.0, 0, 20.0],
             'fractal_dim':[2.0, 0, 5.0],
             'cor_length':[100, 0, 400], },
        'HayterMSA':
            {'name':'HayterMSAStructure',
             'effect_radius':[20, 0, 100],
             'charge':[10, 0, 100],
             'volfraction':[0.2, 0, 1.0],
             'saltconc':[0.0, 0, 1.0],
             'temperature':[400, 100, 800],
             'dielectconst':[50, 0, 200], },
        'Parallelepiped':
            {'name':'ParallelepipedModel',
             'short_a':[20.0, 0, 50.0],
             'short_b':[40.0, 0, 100.0],
             'long_c':[100, 0, 400], },
        'PearlNecklace':
            {'name':'PearlNecklaceModel',
             'radius':[20.0, 0, 50.0],
             'edge_separation':[40.0, 0, 400.0],
             'thick_string':[5, 0, 10],
             'num_pearls':[3, 0, 20], },
        'StickyHS':
            {'name':'StickyHSStructure',
             'effect_radius':[20.0, 0, 50.0],
             'volfraction':[0.1, 0.0, 1.0],
             'perturb':[0.05, 0.0, 2.0],
             'stickiness':[0.2, 0.0, 2.0], },

        }

def assignSASShapeFunction(recipe, contribution, generator, shape, appendstr):
    ''' Sphere
            radius: radis of sphere
        Ellipsoid:
            radius_a: radius along the rotation axis
            radius_b: radius perpendicular to the rotation axis of the ellipsoid
        TriaxialEllipsoid
            semi_axisA
            semi_axisB
            semi_axisC
        FuzzySphere
            radius: radius of the solid sphere
            fuzziness = the STD of the height of fuzzy interfacial thickness (ie., so-called interfacial roughness)
        HardsphereStructure
            effect_radius
            volfraction
        Cylinder:
            radius: radis of cylinder
            length: length of cylinder
        EllipticalCylinder
            r_minor = the radius of minor axis of the cross section
            r_ratio = the ratio of (r_major /r_minor >= 1)
            length = the length of the cylinder
        HollowCylinder
            core_radius : the radius of core
            radius : the radius of shell
            length : the total length of the cylinder
        BCCrystal:
            dnn: Nearest neighbor distance
            d_factor: Paracrystal distortion factor
            radius: radius of the spheres
        FCCrystal:
            dnn: Nearest neighbor distance
            d_factor: Paracrystal distortion factor
            radius: radius of the spheres
        SCCrystal
            dnn: Nearest neighbor distance
            d_factor: Paracrystal distortion factor
            radius: radius of the spheres
        LamellarFFHG
            t_length : tail length
            h_thickness : head thickness
        Lamellar
            bi_thick : bilayer thickness
        LamellarPCrystal
            thickness : lamellar thickness,
            Nlayers : no. of lamellar layers
            spacing : spacing between layers
            pd_spacing : polydispersity of spacing
        LamellarPS
            delta : bilayer thickness,
            n_plate : # of Lamellar plates
            caille : Caille parameter (<0.8 or <1)
        SquareWellStructure
            effect_radius
            volfraction 
            welldepth
            wellwidth
        BarBell:
            rad_bar: radius of the cylindrical bar,
            len_bar: length of the cylindrical bar,
            rad_bell: radius of the spherical bell,
        BinaryHS: (BinaryHSModel model)
            l_radius : large radius of binary hard sphere
            s_radius : small radius of binary hard sphere
        BinaryHSPSF11: (BinaryHSPSF11Model model)
            l_radius : large radius of binary hard sphere
            s_radius : small radius of binary hard sphere
        CappedCylinder: 
            rad_cyl: radius of the cylinder,
            len_cyl: length of the cylinder,
            rad_cap: radius of the semi-spherical cap,
        DiamCylFunc
            radius:
            length:
        DiamEllipFunc
            radius_a:
            radius_b:
        FlexCylEllipX
            length
            kuhn_length
            radius
            axis_ratio
        FlexibleCylinder
            length
            kuhn_length
            radius
        Fractal
            radius : Block radius
            fractal_dim : Fractal dimension
            cor_length : Correlation Length
        HayterMSAStructure
            effect_radius
            charge 
            volfraction 
            temperature
            saltconc
            dielectconst
        Parallelepiped
            short_a: length of short edge  [A]
            short_b: length of another short edge [A]
            long_c: length of long edge  of the parallelepiped [A]
        PearlNecklace
            num_pearls: number of the pearls
            radius: the radius of a pearl
            edge_separation: the length of string segment; surface to surface
            thick_string: thickness (ie, diameter) of the string
        StickyHS
            effect_radius
            volfraction 
            perturb 
            stickiness
    '''
    clsdata = _sasShapeFunctionMeta[shape]
    clsname = clsdata['name']
    sas = __import__('sans.models.' + clsname, globals(), locals(), [clsname], -1)
    sasclass = getattr(sas, clsname)
    sasname = sasclass.__name__.lower()
    cfcalculator = SASCF(sasname, sasclass())
    contribution.registerCalculator(cfcalculator)

    equationstr = contribution.eqstr
    generatorname = generator.name
    equationstr = equationstr.replace(' ' + generatorname + ' ', ' ( %s * %s ) ' % (sasname, generatorname))
    contribution.setEquation(equationstr)
    varlist = clsdata.keys()
    varlist.remove('name')
    for varname in varlist:
        varobj = getattr(cfcalculator, varname)
        recipe.assignNewVar(varobj, varname + appendstr,
                            varvalue=clsdata[varname],
                            tag='shape', p0first=True)
    return
