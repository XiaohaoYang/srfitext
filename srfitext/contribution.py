#!/usr/bin/env python
"""NPContribution class. 

This is a custom FitContribution that simplifies the creation of PDF fits.

"""
import numpy as np

from diffpy.srfit.fitbase import FitContribution
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFContribution

from srfitext.generator import PDFGeneratorExt

class PDFContributionExt(PDFContribution):

    _params = {}
    _extparams = {}

    def __init__(self, name):
        '''Create the NPContribution.
        '''
        super(PDFContributionExt, self).__init__(name)
        # qmax and qmin
        self.newParameter("qmin", 1.0)
        self.newParameter("qmax", 25.0)
        return

    # Data methods

    # Phase methods

    def addStructure(self, stru, name=None, mode=None, parallel=1):
        ''''Add a phase that goes into the PDF calculation.
        name of generator and periodic is determined according to stru if not specfied
        mode is either 'pdf' or 'debye'
        
        name of adapted structure is 'phase'
        '''
        name = stru.name if name == None else name

        if mode == None:
            if stru.periodic:
                mode = 'pdf'
            else:
                mode = 'debye'

        # Based on periodic, create the proper generator.
        gen = PDFGeneratorExt(name, mode)
        # Set up the generator
        gen.setStructure(stru, "phase")
        if parallel > 1:
            gen.parallel(parallel)
        self._setupGenerator(gen)

        return gen.phase

    def _setupGenerator(self, gen):
        """Setup a generator.
        """
        # Add the generator to this FitContribution
        self.addProfileGenerator(gen)
        # Set the proper equation for the fit, depending on the number of
        # phases we have.
        gnames = self._generators.keys()
        eqstr = " + ".join(gnames)
        eqstr = "scale * ( %s )" % eqstr
        self.eqstr = eqstr
        self.setEquation(eqstr)
        # Update with our metadata
        gen.meta.update(self._meta)
        gen.processMetaData()

        # Constrain the shared parameters
        self.constrain(gen.qdamp, self.qdamp)
        self.constrain(gen.qbroad, self.qbroad)
        self.constrain(gen.qmax, self.qmax)

        # constrain qmin if in debye mode
        if gen.mode == 'debye':
            self.constrain(gen.qmin, self.qmin)
        return

    def setParallel(self, parallel=2):
        '''set number of cpus in parallel calculation
        '''
        for gen in self._generators.values():
            gen.parallel(parallel)
        return

    def setData(self, data):
        """Load the data in various formats.

        param data: string or An open file-like object, or list of array or 2D array
            if string or open file-like object, (name of a file that contains
                data or a string containing the data.) this uses the PDFParser
                to load the data and then passes it to the build-in profile with
                loadParsedData.
            if list of array or (2D array), then the first row profile[0] will
                be x, the second row profile[1] will be y and third row profile[2]
                (if exists) will be dy 

        """
        if isinstance(data, (np.ndarray, list)):
            if len(data) == 2:
                x, y = data
                dy = None
            elif len(data) == 3:
                x, y, dy = data
            self.profile.setObservedProfile(xobs=x, yobs=y, dyobs=dy)
        else:
            self.loadData(data)
            # Get the data into a string
            from diffpy.srfit.util.inpututils import inputToString
            datstr = inputToString(data)

            # Load data with a PDFParser
            from diffpy.srfit.pdf.pdfparser import PDFParser
            parser = PDFParser()
            parser.parseString(datstr)

            # Pass it to the profile
            self.profile.loadParsedData(parser)
        return

#
# End of file
