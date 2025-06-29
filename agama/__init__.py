from ._agama import *  # import everything from the C++ library
from ._agama import __doc__, __version__  # these two are not automatically imported
from . import schwarzlib
from .nemofile import NemoFile  # import this one class and merge it into the main namespace
from .pygama import *  # and everything from the Python extension submodule

try:
    import agamacolormaps

    del agamacolormaps  # remove submodule from the namespace
except Exception:
    pass  # no error in case this fails

del schwarzlib

