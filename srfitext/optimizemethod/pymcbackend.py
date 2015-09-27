"""NumPy array trace backend

Store sampling values in memory as a NumPy array.
"""
import numpy as np
from pymc3.backends import base, NDArray
import tables as tb


class HDF5file(base.BaseTrace):
    """HDF5 trace object

    Parameters
    ----------
    name : str
        Name of backend. This has no meaning for the NDArray backend.
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """
    def __init__(self, name=None, model=None, vars=None, filename='mcmc.h5', loadfromfile=False):
        super(HDF5file, self).__init__(name, model, vars)
        self.draws = None
        self.samples = {}
        
        self.filename = filename
        self.loadfilefile = loadfromfile
        return
    
    # # Sampling methods

    def setup(self, draws, chain):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        self.chain = chain
        
        if self.loadfilefile:
            # load h5 file and Concatenate new array if chain is already present
            if os.path.exists(self.filename):
                self.dbfile = tb.open_file(self.filename, mode="r+", title="MCMC Database")
                self.dbsamples = self.dbroot.dbsamples
                old_draws = len(self)
                self.draws = old_draws + draws
                self.draws_idx = old_draws
                if self.draws > old_draws * 100:
                    # create new array and replace old one
                    for varname, shape in self.var_shapes.items():
                        self.dbfile.create_earray(self.dbsamples, 'temp', tb.Float64Atom(), (0,) + shape, expectedrows=draws)
                        self.dbsamples.temp[:draws] = getattr(self.dbsamples, varname)[:]
                        self.dbsamples.remove_node(self.dbsamples, varname)
                        self.dbsamples.temp.rename(varname)
                        self.samples[varname] = getattr(self.dbsamples, varname)
                else:
                    for varname, shape in self.var_shapes.items():
                        self.samples[varname] = getattr(self.dbsamples, varname)
                self.dbfile.flush()
            else:
                raise ValueError('File not existed, check the filename please!')
        else:  # Otherwise, make array of zeros for each variable.
            self.dbfile = tb.open_file(self.filename, mode="w", title="MCMC Database")
            filters = tb.Filters(complevel=5, complib='zlib')
            self.dbsamples = self.dbfile.create_group('/', name='dbsamples', title='MCMC samples', filters=filters)
            
            self.draws = draws
            for varname, shape in self.var_shapes.items():
                self.samples[varname] = \
                    self.dbfile.create_earray(self.dbsamples, varname, tb.Float64Atom(), (0,) + shape, expectedrows=draws)
            self.dbfile.flush()
        return

    def record(self, point):
        """Record results of a sampling iteration.

        Parameters
        ----------
        point : dict
            Values mapped to variable names
        """
        for var, value in zip(self.varnames, self.fn(point)):
            self.samples[var].append(value.reshape((1,) + value.shape))
        return

    def close(self):
        self.dbfile.flush()
        return

    # # Selection methods

    def __len__(self):
        if not self.samples:  # `setup` has not been called.
            return 0
        varname = self.varnames[0]
        return self.samples[varname].nrows

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname : str
        burn : int
        thin : int

        Returns
        -------
        A NumPy array
        """
        return self.samples[varname][burn::thin]

    def _slice(self, idx):
        sliced = NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: values[idx]
                          for varname, values in self.samples.items()}
        return sliced

    def point(self, idx):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.
        """
        idx = int(idx)
        return {varname: values[idx]
                for varname, values in self.samples.items()}
