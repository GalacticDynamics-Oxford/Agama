from .agama import *               # import everything from the C++ library
from .agama import __version__, __doc__  # these two are not automatically imported
from .py.pygama import *           # and everything from the Python extension submodule
from .py import schwarzlib         # also import this submodule, but do not merge it into the main namespace
from .py.nemofile import NemoFile  # import this one class and merge it into the main namespace
try:
    from .py import agamacolormaps # initialize submodule and register some custom colormaps for matplotlib
    del agamacolormaps             # remove submodule from the namespace
except: pass                       # no error in case this fails
del agama                          # remove the C++ library from the root namespace
del py.pygama                      # and the same for the Python extension submodule
del py.schwarzlib
del py.nemofile
