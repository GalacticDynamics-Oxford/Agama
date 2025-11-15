helpstr="""
Installation script for the Agama library.
It could be run just as any regular python program,
> python setup.py install --user
(here --user allows it to be installed without admin privilegies, but should not be used with
anaconda, which has its own environment),
or via pip:
> pip install --user .
(here . means that it should take the files from the current folder);
the former approach is more verbose (this could be amended by providing -q flag to setup.py),
but is deprecated starting from Python 3.10; the latter one (with pip) might fail on older systems.
Note that when installing via pip, it may fail to find the prerequisite packages (numpy,setuptools,wheel);
the solution is to disable build isolation (add a command-line argument --no-build-isolation)

The setup script recognizes some environment variables:
CXX  will set the compiler;
CFLAGS  will set additional compilation flags (e.g., a user-defined include path could be provided
by setting CFLAGS=-I/path/to/my/library/include);
LDFLAGS  likewise will set additional link flags (e.g., -L/path/to/my/library/lib -lmylib).

The setup script will attempt to download and compile the required GSL library (if it wasn't found)
and the optional Eigen, CVXOPT and UNSIO libraries
(asking the user for a confirmation, unless command-line arguments include --yes or --assume-yes).
Eigen is header-only so it is only downloaded; UNSIO is downloaded and compiled (we use a customized
fork of the original C++ library hosted at https://github.com/GalacticDynamics-Oxford/unsio);
CVXOPT is a python library, so it will be installed by calling "pip install [--user] cvxopt"
(you might wish to manually install it if your preferred options are different).
The optional GLPK library is ignored if not found (it is superseded by CVXOPT for all practical
purposes).

In case of problems with automatic downloading of missing libraries (which may occur when using
a proxy server), or if you want to provide them manually, create a subdirectory "extras/"
and place header files in "extras/include" and libraries in "extras/lib".

In the end, the python setup script creates the file Makefile.local and runs make to build the C++
shared library and example programs. If you only need this, you may run python setup.py build_ext
"""

# the setuptools module is not used directly, but it does some weird stuff to distutils;
# if it is loaded mid-way into the installation, things may get wrong, so we import it beforehand
try: import setuptools
except: pass
import os, sys, platform, subprocess, ssl, zipfile, distutils, distutils.core, distutils.dir_util, distutils.file_util
try:        from urllib.request import urlretrieve  # Python 3
except ImportError: from urllib import urlretrieve  # Python 2
from distutils import sysconfig
from distutils.errors import CompileError
from distutils.command.build_ext import build_ext as CmdBuildExt
from distutils.cmd import Command
from distutils.ccompiler import new_compiler

ROOT_DIR   = os.getcwd()  # the directory from which the script was called
EXTRAS_DIR = 'extras'     # this directory will store any third-party libraries installed in the process
compiler = new_compiler() # determine the default compiler and its flags
MSVC = compiler.compiler_type == 'msvc'
uname = platform.uname()
MACOS = uname[0] == 'Darwin'

# retrieve a system configuration variable with the given key, or return an empty string (instead of None)
def get_config_var(key):
    result = sysconfig.get_config_var(key)
    return result if result else ''

# remove file if it exists, or no-op otherwise
def tryRemove(filename):
    try:    os.remove(filename)
    except: pass

# force printing to the terminal even if stdout was redirected
def say(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    if not sys.stdout.isatty():
        # output was redirected, but we still try to send the message to the terminal
        try:
            with open('/dev/tty' if not MSVC else 'CONOUT$', 'w') as out:
                out.write(text)
                out.flush()
        except:
            # /dev/tty may not exist or may not be writable!
            pass

# asking a yes/no question and parse the answer (raise an exception in case of ambiguous answer)
def ask(q):
    say(q if q.endswith('\n') else q+'\n')
    if forceYes:
        result='y'
        say('y\n')
    else:
        result=sys.stdin.readline()
    if not sys.stdin.isatty() and not result:
        # if the input was redirected and no answer was provided,
        # try to read it directly from the true terminal, if it is available
        try:
            with open('/dev/tty' if not MSVC else 'CONIN$', 'r') as stdin:
                result=stdin.readline()
            sys.stdout.write(result)  # and duplicate the entered text to stdout, for posterity
        except:
            result=''  # no reply => will produce an error
    result = result.rstrip()
    if not result:
        raise ValueError('No valid response (y/n) was provided.\n'
        'If you see this message and did not have a chance to enter the response at all (e.g. when using pip or '
        'other non-interactive installation procedure), try running the setup.py script with the --yes flag to provide '
        'automatic affirmative responses to all questions. If installing via pip, this flag can be passed as '
        '--config-settings --build-option=--yes, or in older versions, as --install-option=--yes')
    return distutils.util.strtobool(result)   # this may still fail if the result is not boolean

# get the list of all files in the given directories (including those in nested directories)
def allFiles(*paths):
    return [os.path.join(dirpath, f) \
        for path in paths \
        for dirpath, dirnames, files in os.walk(path) \
        for f in files]

# remove duplicate entries from the list (naive O(N^2) approach, good enough for our needs)
def compressList(src):
    result=[]
    for i in src:
        if not i in result: result += [i]
    return result

def quoteIfSpaces(text):
    return '"%s"' % text  if ' ' in text  else  text

# find out which required and optional 3rd party libraries are present, and create Makefile.local
def createMakefile():

    say('\n    ==== Checking supported compiler options and available libraries ====\n\n')
    # determine the C++ compiler and any externally provided flags:
    try: COMPILE_FLAGS_ALL = os.environ['CFLAGS'].split()
    except KeyError: COMPILE_FLAGS_ALL = []
    try: LINK_FLAGS_ALL = os.environ['LDFLAGS'].split()
    except KeyError: LINK_FLAGS_ALL = []
    # default compilation flags for both the shared library and all example programs that use it
    if MSVC:
        COMPILE_FLAGS_ALL += ['/W3', '-wd4244', '-wd4267', '/O2', '/EHsc', '/nologo']
        LINK_DIR_FLAG = '/LIBPATH:'  # prefix for library search path
        COMPILE_OUT_FLAG = '/Fe:'    # prefix for output file
        COMPILE_WARNING_FATAL_FLAG = '/WX'  # treat warnings as errors
    else:
        COMPILE_FLAGS_ALL += ['-fPIC', '-Wall', '-O2']
        LINK_DIR_FLAG = '-L'
        COMPILE_OUT_FLAG = '-o'
        COMPILE_WARNING_FATAL_FLAG = '-Werror'
    # additional compilation and linking flags
    COMPILE_FLAGS_LIB = []  # compilation of the shared library only (paths to Python, GSL and other third-party libs)
    LINK_FLAGS_LIB = []  # linking of the shared library (path to libpythonXX.so, etc.)
    LINK_FLAGS_LIB_AND_EXE_STATIC = []  # linking of the shared library and any executables that use the static library (paths to third-party libs)
    LINK_FLAGS_EXE_SHARED = []  # linking of executables that use the shared library

    # check if a given test code compiles with given flags
    def runCompiler(code='int main(){}\n', flags=None, dest=None):
        with open('test.cpp', 'w') as f: f.write(code)
        if dest  is None: dest  = 'test.out'
        if flags is None: flags = []
        flags = map(quoteIfSpaces, COMPILE_FLAGS_ALL + flags)
        cmd = '%s test.cpp %s %s %s' % (CC, COMPILE_OUT_FLAG, dest, ' '.join(flags))
        print(cmd)
        result = subprocess.call(cmd, shell=True)
        os.remove('test.cpp')
        tryRemove('test.obj')
        if dest=='test.out': tryRemove(dest)
        return result==0

    # check if a given snippet of code
    # a) compiles into a shared library,
    # b) can be linked against and run without missing anyting
    def runCompileShared(code, flags):
        EXE_CODE = 'void run(); int main() { run(); return 42; }\n'
        EXE_NAME = './agamatest.exe'
        LIB_NAME = './agamatest.so'
        try:
            if flags is None: flags = []
            flags += ['-fPIC', '-shared']
            success = runCompiler(code=code, flags=flags, dest=LIB_NAME) \
            and runCompiler(code=EXE_CODE, flags=[LIB_NAME], dest=EXE_NAME) \
            and subprocess.call(EXE_NAME) == 42
        except: success = False
        # cleanup
        tryRemove(LIB_NAME)
        tryRemove(EXE_NAME)
        return success

    try: CC = os.environ['CXX']  # compiler may be set by an environment variable
    except KeyError:  # determine the default compiler
        if MSVC:
            CC = 'cl'
        else:
            CC = compiler.compiler_cxx
            if isinstance(CC, list): CC = CC[0]
            # unfortunately, if CC specifies the C (not C++) compiler, things don't work as expected..
            if   CC=='cc':    CC='c++'
            elif CC=='gcc':   CC='g++'
            elif CC=='clang': CC='clang++'

    if os.path.isdir(EXTRAS_DIR + '/include'):   # if it was created in the previous invocation of setup.py
        COMPILE_FLAGS_ALL += ['-I' + EXTRAS_DIR + '/include']  # it might already contain some useful stuff
    if os.path.isdir(EXTRAS_DIR + '/lib'):
        LINK_FLAGS_ALL += [LINK_DIR_FLAG + EXTRAS_DIR + '/lib']

    # [1a]: check if a compiler exists at all
    if not runCompiler():
        raise CompileError('Could not locate a compiler (set CXX=... environment variable to override)')

    # [1b]: on MacOS, set the machine architecture explicitly to avoid mismatch between x86_64 and arm64 on the Apple Silicon platform
    if MACOS:
        ARCH_FLAG = ['-arch', uname[4]]  # could be x86_64 or arm64
        if runCompiler(flags=ARCH_FLAG):
            COMPILE_FLAGS_ALL += ARCH_FLAG
            LINK_FLAGS_LIB += ARCH_FLAG

    # [1c]: test the -march flag (optional, allows architecture-dependent compiler optimizations)
    if not MSVC and not MACOS:
        ARCH_FLAG = ['-march=native']
        if runCompiler(flags=ARCH_FLAG):
            COMPILE_FLAGS_ALL += ARCH_FLAG

    # [1d]: test if OpenMP is supported (optional but highly recommended)
    OMP_FLAG = '-fopenmp' if not MSVC else '/openmp'
    OMP_FLAGS= ([OMP_FLAG, COMPILE_WARNING_FATAL_FLAG] + (['-Wno-unknown-pragmas'] if not MSVC else []) +
        COMPILE_FLAGS_LIB + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC)
    OMP_CODE = ('#include <omp.h>\n#include <iostream>\nint main(){int nthreads=0;\n'
        '#pragma omp parallel\n{\n#pragma omp atomic\n++nthreads;\n}\n'
        'std::cout<<"Number of threads: "<<nthreads<<"="<<omp_get_max_threads()<<"\\n";return nthreads;}\n')
    if runCompiler(code=OMP_CODE, flags=OMP_FLAGS):
        say('    **** Compiling with OpenMP support ****\n')
        COMPILE_FLAGS_ALL += [OMP_FLAG]
        if not MSVC:
            LINK_FLAGS_LIB_AND_EXE_STATIC += [OMP_FLAG]
    elif MACOS:
        OMP_H_FOUND = False
        OMP_LIB_FOUND = False
        # On MacOS the clang compiler pretends not to support OpenMP, but in fact it does so if we insist.
        # libomp.so/dylib or libgomp.so or libiomp5.dylib must be present in the system for this to work.
        # In some Anaconda installations, though, linking to the system-default libomp.dylib leads to conflicts
        # with Intel's libiomp5.dylib from Anaconda, so we first try to link to the latter explicitly.
        CONDA_ROOT = os.environ.get('CONDA_EXE')  # full path to conda executable; None if not running from within anaconda
        if isinstance(CONDA_ROOT, str) and os.path.isfile(CONDA_ROOT):
            CONDA_ROOT = CONDA_ROOT.replace('/bin/conda', '')  # now points to the root folder of Anaconda installation
        else:
            CONDA_ROOT = None
        CONDA_PREFIX = os.environ.get('CONDA_PREFIX')  # currently selected conda environment folder (or None)
        if CONDA_PREFIX == CONDA_ROOT:
            CONDA_PREFIX = None  # staying in the base conda environment, no need to check the same folder twice
        if not (
            (isinstance(CONDA_ROOT, str) and os.path.realpath(CONDA_ROOT) in os.path.realpath(sys.executable)) or
            (isinstance(CONDA_PREFIX, str) and os.path.realpath(CONDA_PREFIX) in os.path.realpath(sys.executable)) ):
            CONDA_ROOT = CONDA_PREFIX = None  # we are currently not running python from Anaconda, so forget about it
        #print('CONDA_ROOT=%r, CONDA_PREFIX=%r' % (CONDA_ROOT, CONDA_PREFIX))
        HOMEBREW_PREFIX = subprocess.Popen('brew --prefix', shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
        if not os.path.isdir(HOMEBREW_PREFIX):  # should be /usr/local on Intel or /opt/homebrew on ARM
            HOMEBREW_PREFIX = None
        # first find the include path to omp.h (not using any OpenMP-specific code yet)
        OMP_H_FOUND = runCompiler(code='#include <omp.h>\nint main(){}\n')
        if not OMP_H_FOUND:   # omp.h not found in a known default location
            # search in a few standard directories that may need to be explicitly added to include path
            OMP_INCLUDE_PATHS = []
            if CONDA_PREFIX:
                OMP_INCLUDE_PATHS += [CONDA_PREFIX+'/include']
            if CONDA_ROOT:
                OMP_INCLUDE_PATHS += [CONDA_ROOT+'/include']
            if HOMEBREW_PREFIX:
                OMP_INCLUDE_PATHS += [HOMEBREW_PREFIX+'/opt/libomp/include']
            if os.path.isdir('/opt/local/include/libomp'):  # macports
                OMP_INCLUDE_PATHS += ['/opt/local/include/libomp']
            for OMP_INCLUDE_PATH in OMP_INCLUDE_PATHS:
                if os.path.isfile(OMP_INCLUDE_PATH+'/omp.h') \
                    and runCompiler(code='#include <omp.h>\nint main(){}\n', flags=['-I'+OMP_INCLUDE_PATH]):
                    COMPILE_FLAGS_ALL += ['-I'+OMP_INCLUDE_PATH]
                    OMP_H_FOUND = True
                    break
        # then find the location of OpenMP dynamic library
        if OMP_H_FOUND:
            OMP_LIB_PATHS = []
            # first try Anaconda's libiomp5 if possible
            if CONDA_PREFIX and os.path.isfile(CONDA_PREFIX+'/lib/libiomp5.dylib'):
                OMP_LIB_PATHS += [CONDA_PREFIX+'/lib/libiomp5.dylib']
            if CONDA_ROOT and os.path.isfile(CONDA_ROOT+'/lib/libiomp5.dylib'):
                OMP_LIB_PATHS += [CONDA_ROOT+'/lib/libiomp5.dylib']
            # then look for libgomp or libomp under default system-wide paths
            OMP_LIB_PATHS += ['-lgomp', '-lomp']
            if HOMEBREW_PREFIX:
                OMP_LIB_PATHS += ['-L'+HOMEBREW_PREFIX+'/opt/libomp/lib -lomp']
            if os.path.isdir('/opt/local/lib/libomp'):  # macports
                OMP_LIB_PATHS += ['-L/opt/local/lib/libomp -lomp']
            EXE_NAME = './agamatest.exe'
            for OMP_LIB_PATH in OMP_LIB_PATHS:
                if runCompiler(code=OMP_CODE, flags=[OMP_LIB_PATH, '-Xpreprocessor'] + OMP_FLAGS, dest=EXE_NAME):
                    # ensuring that the code compiles and links is not enough, need to check if it can be run!
                    # (the snag is that the OpenMP library may not be in LD_LIBRARY_PATH so not found at runtime)
                    OMP_LIB_FOUND = subprocess.call(EXE_NAME) >= 0
                    if OMP_LIB_FOUND:
                        COMPILE_FLAGS_ALL += ['-Xpreprocessor', OMP_FLAG]
                        LINK_FLAGS_ALL += [OMP_LIB_PATH]
                        break
                    elif not OMP_LIB_PATH.startswith('-L') and runCompiler(code=OMP_CODE,
                            flags=['-Wl,-rpath,' + OMP_LIB_PATH[:OMP_LIB_PATH.rfind('/')],
                            OMP_LIB_PATH, '-Xpreprocessor'] + OMP_FLAGS, dest=EXE_NAME):
                        # failed to run: may need to add a '-rpath' argument to the linker
                        OMP_LIB_FOUND = subprocess.call(EXE_NAME) >= 0
                        if OMP_LIB_FOUND:
                            COMPILE_FLAGS_ALL += ['-Xpreprocessor', OMP_FLAG]
                            LINK_FLAGS_ALL += ['-Wl,-rpath,' + OMP_LIB_PATH[:OMP_LIB_PATH.rfind('/')], OMP_LIB_PATH]
                            break
            tryRemove(EXE_NAME)

        if OMP_LIB_FOUND:
            say('    **** Compiling with OpenMP support ****\n')
        elif not ask('Warning, OpenMP is not supported\n'+
            "If you're compiling on MacOS with clang, you'd better install another compiler such as GCC, "
            "or if you're using homebrew, install libomp via 'brew install libomp' and retry running this setup script\n"+
            'Do you want to continue without OpenMP? [Y/N] '):
            exit(1)

    if not MSVC:
        # [1e]: test if C++11 is supported (optional)
        CXX11_FLAG = '-std=c++11'
        if runCompiler(flags=[CXX11_FLAG]):
            COMPILE_FLAGS_ALL += [CXX11_FLAG]
        else:
            # check if we can suppress the following pre-C++11 warning
            WARNING_FLAG = '-Wno-invalid-offsetof'
            if runCompiler(flags=[COMPILE_WARNING_FATAL_FLAG, WARNING_FLAG]):
                COMPILE_FLAGS_ALL += [WARNING_FLAG]

        # [1f]: special treatment for Intel compiler to restore determinism in OpenMP-parallelized loops
        INTEL_FLAG = '-qno-opt-dynamic-align'
        if runCompiler(code='#ifndef __INTEL_COMPILER\n#error\n#endif\nint main(){}\n', flags=[INTEL_FLAG]):
            COMPILE_FLAGS_ALL += [INTEL_FLAG]

        # [1g]: remove a couple of low-importance warnings that are unavoidably generated
        # by the Python C API conventions (only if they are supported by the compiler)
        for WARNING_FLAG in ['-Wno-missing-field-initializers', '-Wno-cast-function-type']:
            if runCompiler(flags=[COMPILE_WARNING_FATAL_FLAG, WARNING_FLAG]):
                COMPILE_FLAGS_ALL += [WARNING_FLAG]

    # [2a]: check that NumPy is present (required by the python interface)
    try:
        import numpy
        NUMPY_INC = '-I' + numpy.get_include()
    except ImportError:
        raise CompileError('NumPy is not present - python extension cannot be compiled.\n'
            'If you see this error when installing via pip, and are sure that your python environment actually '
            'has numpy installed, you may need to run pip with a command-line argument --no-build-isolation, '
            'and make sure that setuptools and wheel packages are also installed.')

    # [2b]: find out the paths to Python.h and libpythonXX.{a,so,dylib,...} (this is rather tricky)
    # and all other relevant compilation/linking flags needed to build a shared library that uses Python
    PYTHON_INC = '-I' + sysconfig.get_python_inc()
    if not MSVC and not runCompiler(code="#include <Python.h>\nint main(){}\n", flags=[PYTHON_INC]):
        raise CompileError('Python.h is not found.\n'
            'You may need to install the package "python-dev", '
            '"python'+sysconfig.get_config_var('VERSION')+'-dev", "python-devel", or something similar '
            'using the system package manager (apt-get, yum, or whatever is appropriate for your system).\n' +
            'Alternatively, you may install Anaconda in place of your system-wide Python.')

    # various other system libraries that are needed at link time
    PYTHON_LIB_EXTRA = compressList(
        get_config_var('LIBS').split() +
        get_config_var('SYSLIBS').split())

    # test code for a shared library (Python extension module)
    if sys.version_info[0]*0x100 + sys.version_info[1] < 0x303:
        sys_prefix_var = 'sys.prefix'
        sys_prefix_val =  sys.prefix
    else:
        # starting from version 3.3, sys.prefix may be modified if running in venv,
        # while base_prefix keeps the original value (directory where the python is installed)
        sys_prefix_var = 'sys.base_prefix'
        sys_prefix_val =  sys.base_prefix
    PYTEST_LIB_CODE = """
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
void bla() {PyRun_SimpleString("import sys, os; print('sys.prefix='+os.path.realpath(%s));");}
void run() {Py_Initialize();bla();Py_Finalize();}
PyMODINIT_FUNC
""" % sys_prefix_var
    if sys.version_info[0]==2:  # Python 2.6-2.7
        PYTEST_LIB_CODE += """
initagamatest(void) {
    Py_InitModule3("agamatest", NULL, "doc");
    import_array();
    bla();
}
"""
    else:  # Python 3.x
        PYTEST_LIB_CODE += """
PyInit_agamatest(void) {
    static PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "agamatest", "doc", -1, NULL};
    PyObject* mod = PyModule_Create(&moduledef);
    import_array1(mod);
    bla();
    return mod;
}
"""
    # try compiling a test code with the provided link flags (in particular, the name of Python library):
    # check that a sample C++ program with embedded python compiles, links and runs properly
    def tryPythonCode(PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS=[]):
        print('    **** Trying the following options for linking against Python library ****')
        # test code for a program that loads this shared library
        PYTEST_EXE_CODE = 'extern void run();int main(){run();}\n'
        PYTEST_LIB_NAME = './agamatest.so'
        PYTEST_EXE_NAME = './agamatest.exe'
        # try compiling the test shared library
        if not (runCompiler(code=PYTEST_LIB_CODE,
                flags=[PYTHON_INC, NUMPY_INC, '-shared', '-fPIC'] + PYTHON_SO_FLAGS +
                LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC,
                dest=PYTEST_LIB_NAME) and os.path.isfile(PYTEST_LIB_NAME) ):
            return False  # the program couldn't be compiled at all (try the next variant)
        # if succeeded, compile the test program that uses this library
        if not (runCompiler(code=PYTEST_EXE_CODE,
                flags=[PYTEST_LIB_NAME] + PYTHON_EXE_FLAGS + LINK_FLAGS_ALL + LINK_FLAGS_EXE_SHARED,
                dest=PYTEST_EXE_NAME) and os.path.isfile(PYTEST_EXE_NAME) ):
            return False  # can't find compiled test program
        resultexe = subprocess.Popen(PYTEST_EXE_NAME,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
        # the test program might not be able to find the python home, in which case manually provide it
        if 'Could not find platform independent libraries <prefix>' in resultexe:
            resultexe = subprocess.Popen(PYTEST_EXE_NAME,
                env=dict(os.environ, PYTHONHOME=sys.prefix),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
        # also try loading this shared library as an extension module
        procpy = subprocess.Popen('"%s" -c "import agamatest"' % sys.executable, shell=True, \
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        resultpy = procpy.communicate()[0].decode().rstrip()
        returnpy = procpy.returncode
        # clean up
        os.remove(PYTEST_EXE_NAME)
        os.remove(PYTEST_LIB_NAME)
        # check if the results (reported library path prefix) are the same as we have in this script
        # (test for substring because the output may contain additional messages)
        sysprefix = 'sys.prefix=' + os.path.realpath(sys_prefix_val)
        if sysprefix not in resultexe  or  sysprefix not in resultpy:
            print("Test program doesn't seem to use the same version of Python, "+\
                "or the library path is reported incorrectly: \n"+\
                "Expected: "+sysprefix+"\n"+\
                "Received: "+resultexe+"\n"+\
                "From py:  "+resultpy+('' if returnpy==0 else ' (crashed with error '+str(returnpy)+')'))
            return False
        print('    **** Successfully linked using these options ****')
        return True   # this combination of options seems reasonable...

    # determine compilation and linking options for the python extension module when using MSVC on Windows
    def findPythonLibMSVC():
        PYTHON_SO_FLAGS = ['/LIBPATH:%s' % os.path.join(sys.exec_prefix, 'libs')]
        PYTHON_EXE_FLAGS= []
        if runCompiler(code=PYTEST_LIB_CODE,
            flags=[PYTHON_INC, NUMPY_INC, '/link', '/dll', '/out:agamatest.pyd'] + PYTHON_SO_FLAGS):
            # test the compiled extension module
            procpy = subprocess.Popen('"%s" -c "import agamatest"' % sys.executable, shell=True, \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            resultpy = procpy.communicate()[0].decode().rstrip()
            returnpy = procpy.returncode
            os.remove('agamatest.pyd')   # various cleanups
            tryRemove('agamatest.lib')
            tryRemove('agamatest.exp')
            # check if the results (reported library path prefix) are the same as we have in this script
            sysprefix = 'sys.prefix=' + os.path.realpath(sys.prefix)
            if returnpy == 0 and sysprefix in resultpy:  # everything fine
                return PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS
            print("Test program doesn't seem to use the same version of Python, "+\
                "or the library path is reported incorrectly: \n"+\
                "Expected: "+sysprefix+"\n"+\
                "Received: "+resultpy+('' if returnpy==0 else ' (crashed with error '+str(returnpy)+')'))
        raise CompileError('Could not compile test program which uses libpython' +
            sysconfig.get_config_var('VERSION'))

    # explore various possible combinations of file name and path to the python library on Linux/MacOS
    def findPythonLib():
        # try linking against the static python library libpython**.a, if this does not succeed,
        # try the shared library libpython**.so** or libpython**.dylib
        PY_VERSION = sysconfig.get_config_var('VERSION') + getattr(sys, 'abiflags', '')
        for PYTHON_LIB_FILENAME in compressList([sysconfig.get_config_var(x) for x in ['LIBRARY', 'LDLIBRARY', 'INSTSONAME']] +
            ['libpython%s.a' % PY_VERSION, 'libpython%s.so' % PY_VERSION, 'libpython%s.dylib' % PY_VERSION] ):
            for PYTHON_LIB_PATH in compressList([sysconfig.get_config_var(x) for x in ['LIBPL', 'LIBDIR', 'srcdir']]):
                # obtain full path to the python library
                PYTHON_LIB_FILEPATH = os.path.join(PYTHON_LIB_PATH, PYTHON_LIB_FILENAME)
                # check if the file exists at all at the given location
                if os.path.isfile(PYTHON_LIB_FILEPATH):
                    # flags for compiling the shared library which serves as a Python extension module
                    PYTHON_SO_FLAGS = [PYTHON_LIB_FILEPATH] + PYTHON_LIB_EXTRA
                    # other libraries depend on whether this is a static or a shared python library
                    if PYTHON_LIB_FILENAME.endswith('.a') and not sysconfig.get_config_var('PYTHONFRAMEWORK'):
                        PYTHON_SO_FLAGS += get_config_var('LINKFORSHARED').split()
                    # the stack_size flag is problematic and needs to be removed
                    PYTHON_SO_FLAGS = [x for x in PYTHON_SO_FLAGS if not x.startswith('-Wl,-stack_size,')]
                    if tryPythonCode(PYTHON_SO_FLAGS):
                        return PYTHON_SO_FLAGS, []   # successful compilation
                    elif not PYTHON_LIB_FILENAME.endswith('.a'):
                        # sometimes the python installation is so wrecked that the linker can find and use
                        # the shared library libpython***.so, but this library is not in LD_LIBRARY_PATH and
                        # cannot be found when loading the python extension module outside python itself.
                        # the (inelegant) fix is to hardcode the path to this libpython***.so as -rpath.
                        RPATH = ['-Wl,-rpath,'+PYTHON_LIB_PATH]
                        # This rpath can already be in LINK_FLAGS_ALL due to the OpenMP setup, so don't put it twice
                        if RPATH[0] in LINK_FLAGS_ALL:
                            RPATH = []
                        else:  # extend the linker options and try again
                            print('Trying rpath')
                            if tryPythonCode(PYTHON_SO_FLAGS + RPATH):
                                return PYTHON_SO_FLAGS + RPATH, []
                        # another attempt with a hardcoded path
                        MACOS_RPATH = '/Library/Developer/CommandLineTools/Library/Frameworks/'
                        if os.path.isdir(MACOS_RPATH) and tryPythonCode(PYTHON_SO_FLAGS + ['-Wl,-rpath,'+MACOS_RPATH]):
                            return PYTHON_SO_FLAGS + ['-Wl,-rpath,'+MACOS_RPATH], []
                        if '-undefined dynamic_lookup' in sysconfig.get_config_var('LDSHARED'):
                            print('Trying the last resort solution')  # relevant for Anaconda installations
                            PYTHON_SO_FLAGS = ['-undefined', 'dynamic_lookup'] + PYTHON_LIB_EXTRA
                            PYTHON_EXE_FLAGS = RPATH + [PYTHON_LIB_FILEPATH]
                            if tryPythonCode(PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS):
                                return PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS
                            if RPATH and tryPythonCode(PYTHON_SO_FLAGS + RPATH, PYTHON_EXE_FLAGS):
                                return PYTHON_SO_FLAGS + RPATH, PYTHON_EXE_FLAGS

        # if the above efforts did not succeed, try the options suggested by python-config
        PYTHON_CONFIG = 'python%s-config --ldflags' % sysconfig.get_config_var('VERSION')
        print('    **** Trying the options provided by %s ****' % PYTHON_CONFIG)
        sub = subprocess.Popen(PYTHON_CONFIG, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        PYTHON_SO_FLAGS = sub.communicate()[0].decode().rstrip().split()
        if sub.returncode==0 and tryPythonCode(PYTHON_SO_FLAGS):
            return PYTHON_SO_FLAGS, []

        # check if the error is due to ARM/x86_64 incompatibility on Apple M1
        # (this should be fixed by [1d] - specifying "-arch x86_64" or "-arch arm64" instead of "-march=native" ?)
        if MACOS and 'ARM64' in uname[3]:
            raise CompileError("If you see an error 'ld: symbol(s) not found for architecture arm64', " +
                "it may be because you are running a x86_64 version of Python " +
                "on an ARM (Apple Silicon/M1) machine, and the Python extension module " +
                "compiled for the ARM architecture is incompatible with the Python interpreter. " +
                "The solution is to install a native ARM Python version (e.g., Anaconda 2022.05 or newer)")
        # if none of the above combinations worked, give up...
        raise CompileError('Could not compile test program which uses libpython' +
            sysconfig.get_config_var('VERSION'))

    # [2c]: find the python library and other relevant linking flags
    PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS = findPythonLib() if not MSVC else findPythonLibMSVC()
    COMPILE_FLAGS_LIB     += ['-DHAVE_PYTHON', PYTHON_INC, NUMPY_INC]
    LINK_FLAGS_LIB        += PYTHON_SO_FLAGS
    LINK_FLAGS_EXE_SHARED += PYTHON_EXE_FLAGS

    # [3]: check that GSL is present, and find out its version (required)
    # try compiling a snippet of code into a shared library (tests if GSL has been compiled with -fPIC),
    # then compiling a test program that loads this library (tests if the correct version of GSL is loaded at link time)
    GSL_TEST_CODE = """#include <gsl/gsl_version.h>
    #if !defined(GSL_MAJOR_VERSION) || (GSL_MAJOR_VERSION == 1) && (GSL_MINOR_VERSION < 15)
    #error "GSL version is too old (need at least 1.15)"
    #endif
    #include <gsl/gsl_integration.h>
    void run() { gsl_integration_cquad_workspace_alloc(10); }
    """
    # prevent "certificate verify failed" error in urlretrieve
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass
    if not MSVC:   # install GSL on Linux/MacOS
        if runCompileShared(GSL_TEST_CODE, flags=COMPILE_FLAGS_LIB +
                LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC + ['-lgsl', '-lgslcblas']):
            # apparently the headers and libraries can be found in some standard location, so we only list their names
            LINK_FLAGS_LIB_AND_EXE_STATIC += ['-lgsl', '-lgslcblas']
        else:
            if not ask('GSL library (required) is not found\n'+
                'Should we try to download and compile it now? [Y/N] '): exit(1)
            distutils.dir_util.mkpath(EXTRAS_DIR)
            os.chdir(EXTRAS_DIR)
            say("Downloading GSL\n")
            filename = 'gsl.tar.gz'
            dirname  = 'gsl-2.8'
            try:
                urlretrieve('https://ftpmirror.gnu.org/gnu/gsl/gsl-2.8.tar.gz', filename)
                if os.path.isfile(filename):
                    say('Unpacking GSL\n')
                    subprocess.call(['tar', '-zxf', filename])    # unpack the archive
                    os.remove(filename)  # remove the downloaded archive
                    if not os.path.isdir(dirname): raise Exception('Error unpacking GSL')
                else: raise RuntimeError('Cannot find downloaded file')
            except Exception as e:
                raise CompileError(str(e) + '\nError downloading GSL library, aborting...\n'+
                'You may try to manually compile GSL and install it to '+ROOT_DIR+'/'+EXTRAS_DIR+', so that '+
                'the header files are in '+EXTRAS_DIR+'/include and library files - in '+EXTRAS_DIR+'/lib')
            say('Compiling GSL (may take a few minutes)\n')
            result = subprocess.call('(cd '+dirname+'; ./configure --prefix='+os.getcwd()+
                ' CFLAGS="-fPIC -O2" --enable-shared=no; make; make install) > gsl-install.log', shell=True)
            if result != 0 or not os.path.isfile('lib/libgsl.a'):
                 raise CompileError('GSL compilation failed (check '+EXTRAS_DIR+'/gsl-install.log)')
            distutils.dir_util.remove_tree(dirname)  # clean up source and build directories
            COMPILE_FLAGS_LIB  += ['-I'+EXTRAS_DIR+'/include']
            LINK_FLAGS_LIB_AND_EXE_STATIC += [EXTRAS_DIR+'/lib/libgsl.a', EXTRAS_DIR+'/lib/libgslcblas.a']
            os.chdir(ROOT_DIR)
    else:  # install a GSL port on Windows with MSVC
        if runCompiler(GSL_TEST_CODE + 'int main() {run();}', flags=COMPILE_FLAGS_LIB +
                ['/link', 'gsl.lib', 'cblas.lib'] + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC):
            LINK_FLAGS_LIB_AND_EXE_STATIC += ['gsl.lib', 'cblas.lib']
        else:
            if not ask('GSL library (required) is not found\n'+
                'Should we try to download and compile it now? [Y/N] '): exit(1)
            distutils.dir_util.mkpath(EXTRAS_DIR)
            os.chdir(EXTRAS_DIR)
            say("Downloading GSL\n")
            dirname  = 'gsl-master'
            filename = dirname + '.zip'
            try:
                urlretrieve('http://agama.software/files/%s' % filename, filename)
                if os.path.isfile(filename):
                    say('Unpacking GSL\n')
                    zipf = zipfile.ZipFile(filename, 'r')  # unpack the archive
                    zipf.extractall()
                    zipf.close()
                    os.remove(filename)  # remove the downloaded archive
                    if not os.path.isdir(dirname): raise Exception('Error unpacking GSL')
                else: raise RuntimeError('Cannot find downloaded file')
            except Exception as e:
                raise CompileError(str(e) + '\nError downloading GSL library, aborting...\n'+
                'You may try to manually compile GSL and install it to '+ROOT_DIR+'/'+EXTRAS_DIR+', so that '+
                'the header files are in '+EXTRAS_DIR+'/include and library files - in '+EXTRAS_DIR+'/lib')
            say('Compiling GSL (may take a few minutes)\n')
            os.chdir(dirname+'\\build.vc')
            result = subprocess.call('msbuild /v:n gslhdrs  /p:Configuration=Release >>../../gsl-install.log', shell=True)
            result+= subprocess.call('msbuild /v:n cblaslib /p:Configuration=Release >>../../gsl-install.log', shell=True)
            result+= subprocess.call('msbuild /v:n gsllib   /p:Configuration=Release >>../../gsl-install.log', shell=True)
            if result != 0 or not os.path.isfile('lib/x64/Release/gsl.lib'):
                raise CompileError('GSL compilation failed (check '+EXTRAS_DIR+'/gsl-install.log)')
            # copy the static libraries to extras/lib
            distutils.dir_util.mkpath('../../lib')
            distutils.file_util.copy_file('lib/x64/Release/gsl.lib',   '../../lib/')
            distutils.file_util.copy_file('lib/x64/Release/cblas.lib', '../../lib/')
            distutils.dir_util.copy_tree('../gsl', '../../include/gsl', verbose=False)
            os.chdir('../..')
            # delete the compiled directory
            distutils.dir_util.remove_tree(dirname)  # clean up source and build directories
            COMPILE_FLAGS_LIB += ['-I'+EXTRAS_DIR+'/include']
            LINK_FLAGS_LIB_AND_EXE_STATIC += [LINK_DIR_FLAG+EXTRAS_DIR+'/lib', 'gsl.lib', 'cblas.lib']
            os.chdir(ROOT_DIR)
            if not runCompiler(GSL_TEST_CODE + 'int main() {run();}',
                    flags=COMPILE_FLAGS_LIB + ['/link'] + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC):
                raise CompileError('GSL compilation did not succeed')

    # [4]: test if Eigen library is present (optional)
    if runCompiler(code='#include <Eigen/Core>\nint main(){}\n', flags=COMPILE_FLAGS_LIB):
        COMPILE_FLAGS_LIB += ['-DHAVE_EIGEN']
    else:
        if ask('Eigen library (recommended) is not found\n'+
                'Should we try to download it now (no compilation needed)? [Y/N] '):
            distutils.dir_util.mkpath(EXTRAS_DIR+'/include/unsupported')
            os.chdir(EXTRAS_DIR)
            say('Downloading Eigen\n')
            filename = 'Eigen.zip'
            dirname  = 'eigen-3.4.1'
            try:
                urlretrieve('https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip', filename)
                urlretrieve('https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip', filename)
                if os.path.isfile(filename):
                    say('Unpacking Eigen\n')
                    zipf = zipfile.ZipFile(filename, 'r')
                    zipf.extractall()
                    zipf.close()
                    if os.path.isdir(dirname):
                        distutils.dir_util.copy_tree(dirname+'/Eigen', 'include/Eigen', verbose=False)  # copy the headers
                        distutils.dir_util.copy_tree(dirname+'/unsupported/Eigen', 'include/unsupported/Eigen', verbose=False)
                        distutils.dir_util.remove_tree(dirname)  # and delete the rest
                        COMPILE_FLAGS_LIB += ['-DHAVE_EIGEN', '-I'+EXTRAS_DIR+'/include']
                    os.remove(filename)                          # remove the downloaded archive
                else: raise RuntimeError('Cannot find downloaded file')
            except Exception as e:
                say('Failed to install Eigen: '+str(e)+'\n')     # didn't succeed with Eigen
            os.chdir(ROOT_DIR)

    # [5a]: test if CVXOPT is present (optional); install if needed
    skip_cvxopt = False
    try:
        import cvxopt  # import the python module
    except ImportError:
        if ask('CVXOPT library (needed only for Schwarzschild modelling) is not found\n'
                'Should we try to install it now? [Y/N] '):
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cvxopt'] +
                    (['--user'] if '--user' in sys.argv else []))
                # check if the module was successfully installed and can be imported;
                # but we have to do this in a new python process, since the current one
                # may not be able to retry importing a module that once failed
                result = subprocess.run([sys.executable, '-c', 'import cvxopt'], stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode().strip())
            except Exception as e:
                say('Failed to install CVXOPT: '+str(e)+'\n')
                skip_cvxopt = True
        else:
            skip_cvxopt = True

    # [5b]: if the cvxopt module is available in Python, make sure that we also have C header files
    if not skip_cvxopt:
        if runCompiler(code='#include <cvxopt.h>\nint main(){import_cvxopt();}\n',
                flags=COMPILE_FLAGS_LIB + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC):
            COMPILE_FLAGS_LIB += ['-DHAVE_CVXOPT']
        else:
            # download the C header file if it does not appear to be present in a default location
            distutils.dir_util.mkpath(EXTRAS_DIR+'/include')
            say('Downloading CVXOPT header files\n')
            try:
                urlretrieve('https://raw.githubusercontent.com/cvxopt/cvxopt/master/src/C/cvxopt.h',
                    EXTRAS_DIR+'/include/cvxopt.h')
                urlretrieve('https://raw.githubusercontent.com/cvxopt/cvxopt/master/src/C/blas_redefines.h',
                    EXTRAS_DIR+'/include/blas_redefines.h')
            except: pass  # problems in downloading, skip it
            if  os.path.isfile(EXTRAS_DIR+'/include/cvxopt.h') and \
                os.path.isfile(EXTRAS_DIR+'/include/blas_redefines.h'):
                COMPILE_FLAGS_LIB += ['-DHAVE_CVXOPT', '-I'+EXTRAS_DIR+'/include']
            else:
                say('Failed to download CVXOPT header files, this feature will not be available\n')

    # [6]: test if GLPK is present (optional - ignored if not found)
    if not MSVC and runCompileShared('#include <glpk.h>\nvoid run() { glp_create_prob(); }\n',
            flags=COMPILE_FLAGS_LIB + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC + ['-lglpk']):
        COMPILE_FLAGS_LIB += ['-DHAVE_GLPK']
        LINK_FLAGS_LIB_AND_EXE_STATIC += ['-lglpk']
    else:
        say('GLPK library (optional) is not found, ignored\n')

    # [7]: test if UNSIO is present (optional), download and compile if needed
    if not MSVC and runCompileShared('#include <uns.h>\nvoid run() { uns::CunsOut("temp", "nemo"); }\n',
            flags=COMPILE_FLAGS_LIB + LINK_FLAGS_ALL + LINK_FLAGS_LIB + LINK_FLAGS_LIB_AND_EXE_STATIC + ['-lunsio', '-lnemo']):
        COMPILE_FLAGS_LIB += ['-DHAVE_UNSIO']
        LINK_FLAGS_LIB_AND_EXE_STATIC += ['-lunsio', '-lnemo']
    elif not MSVC:
        if ask('UNSIO library (optional; used for input/output of N-body snapshots) is not found\n'+
                'Should we try to download and compile it now? [Y/N] '):
            distutils.dir_util.mkpath(EXTRAS_DIR)
            distutils.dir_util.mkpath(EXTRAS_DIR+'/include')
            distutils.dir_util.mkpath(EXTRAS_DIR+'/lib')
            say('Downloading UNSIO\n')
            filename = EXTRAS_DIR+'/unsio-master.zip'
            dirname  = EXTRAS_DIR+'/unsio-master'
            try:
                urlretrieve('https://github.com/GalacticDynamics-Oxford/unsio/archive/master.zip', filename)
                if os.path.isfile(filename):
                    subprocess.call('(cd '+EXTRAS_DIR+'; unzip ../'+filename+') >/dev/null', shell=True)  # unpack
                    os.remove(filename)  # remove the downloaded archive
                    say('Compiling UNSIO\n')
                    result = subprocess.call('(cd '+dirname+'; make) > '+EXTRAS_DIR+'/unsio-install.log', shell=True)
                    if result == 0 and os.path.isfile(dirname+'/libnemo.a') and os.path.isfile(dirname+'/libunsio.a'):
                        # successfully compiled: copy the header files to extras/include
                        for hfile in ['componentrange.h', 'ctools.h', 'snapshotinterface.h', 'uns.h', 'userselection.h']:
                            distutils.file_util.copy_file(dirname+'/unsio/'+hfile, EXTRAS_DIR+'/include')
                        # copy the static libraries to extras/lib
                        distutils.file_util.copy_file(dirname+'/libnemo.a',  EXTRAS_DIR+'/lib')
                        distutils.file_util.copy_file(dirname+'/libunsio.a', EXTRAS_DIR+'/lib')
                        # delete the compiled directory
                        distutils.dir_util.remove_tree(dirname)
                        UNSIO_COMPILE_FLAGS = ['-I'+EXTRAS_DIR+'/include']
                        UNSIO_LINK_FLAGS = [EXTRAS_DIR+'/lib/libunsio.a', EXTRAS_DIR+'/lib/libnemo.a']
                        if runCompiler(code='#include <uns.h>\nint main(){}\n', flags=UNSIO_COMPILE_FLAGS + UNSIO_LINK_FLAGS):
                            COMPILE_FLAGS_LIB += ['-DHAVE_UNSIO'] + UNSIO_COMPILE_FLAGS
                            LINK_FLAGS_LIB_AND_EXE_STATIC += UNSIO_LINK_FLAGS
                        else:
                            raise CompileError('Failed to link against the just compiled UNSIO library')
                    else:
                        raise CompileError('Failed compiling UNSIO (check '+EXTRAS_DIR+'/unsio-install.log)')
            except Exception as e:  # didn't succeed with UNSIO
                say(str(e)+'\n')

    # [8]: some OS-dependent wizardry needed to ensure that the executables are linked to
    # the shared library using a relative path, so that it will be looked for in the same
    # folder as the executables (a symlink is created as exe/agama.so -> agama.so).
    # This allows the executables to be copied/moved to and run from any other folder
    # without the need to put agama.so into a system-wide folder such as /usr/local/lib,
    # or add it to LD_LIBRARY_PATH, provided that a copy of the shared library or a symlink
    # to it resides in the same folder as the executable program.
    if MACOS:
        # on MacOS we need to modify the header of the shared library, which is then
        # automatically used to embed the correct relative path into each executable
        LINK_FLAGS_LIB += ['-Wl,-install_name,@executable_path/agama.so']
    elif not MSVC:
        # on Linux we need to pass this flag to every compiled executable in the exe/ subfolder
        LINK_FLAGS_EXE_SHARED += ["-Wl,-rpath,'$$ORIGIN'"]

    # [9]: when a non-C++ executable is linked to the static library agama.a,
    # it should additionally link to the C++ standard library, whose name depends on the compiler
    CLANG = not MSVC and 'clang' in subprocess.Popen(CC + ' -v', shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
    LINK_FLAG_EXE_STATIC_NONCPP = '-lc++' if CLANG else '-lstdc++'

    # [99]: put everything together and create Makefile.local
    with open('Makefile.local','w') as f: f.write(
        ('# set the default compiler if no value is found in the environment variables or among command-line arguments\n' +
        'ifeq ($(origin CXX),default)\nCXX = ' + CC + '\nendif\n' +
        'ifeq ($(origin FC), default)\nFC  = gfortran\nendif\nLINK = $(CXX)\n' if not MSVC else '') +
        '# compilation flags for both the shared library and any programs that use it\n' +
        'COMPILE_FLAGS_ALL = ' + ' '.join(map(quoteIfSpaces, compressList(COMPILE_FLAGS_ALL))) + '\n' +
        '# additional compilation flags for the library files only (src/*.cpp)\n' +
        'COMPILE_FLAGS_LIB = ' + ' '.join(map(quoteIfSpaces, compressList(COMPILE_FLAGS_LIB))) + '\n' +
        '# linking flags for both the shared library (agama.so) and any executable programs, regardless of how they are linked\n' +
        'LINK_FLAGS_ALL = ' + ' '.join(map(quoteIfSpaces, compressList(LINK_FLAGS_ALL))) + '\n' +
        '# additional linking flags for the shared library only (agama.so)\n' +
        'LINK_FLAGS_LIB = ' + ' '.join(map(quoteIfSpaces, compressList(LINK_FLAGS_LIB))) + '\n' +
        '# additional linking flags for the shared library (agama.so) and executables that use the static library (agama.a)\n' +
        'LINK_FLAGS_LIB_AND_EXE_STATIC = ' + ' '.join(map(quoteIfSpaces, compressList(LINK_FLAGS_LIB_AND_EXE_STATIC))) + '\n' +
        '# additional linking flags for executables which use the shared library (agama.so)\n' +
        'LINK_FLAGS_EXE_SHARED = ' + ' '.join(map(quoteIfSpaces, compressList(LINK_FLAGS_EXE_SHARED))) + '\n' +
        '# additional linking flags for non-C++ executables (e.g. fortran programs) that use the static library (agama.a)\n' +
        'LINK_FLAGS_EXE_STATIC_NONCPP = ' + LINK_FLAG_EXE_STATIC_NONCPP + '\n')


# Custom build step that manually creates the makefile and then calls 'make' to create the shared library
class MyBuildExt(CmdBuildExt):
    def run(self):
        # check if Makefile.local is already present
        if not os.path.isfile('Makefile.local') or \
            not ask('Makefile.local already exists, should we use it (Y) or generate a new one (N)? '):
                createMakefile()
        # run custom build step (make)
        say('\n    ==== Compiling the C++ library ====\n\n')
        if MSVC:
            make = 'nmake -f Makefile.msvc'
            sharedname = 'agama.pyd'
            staticname = 'agama.lib'
        else:  # standard unix (including macos)
            make = 'make'
            sharedname = 'agama.so'
            staticname = 'agama.a'
        if subprocess.call(make) != 0 or not os.path.isfile(sharedname):
            raise CompileError("Compilation failed")
        if not os.path.isdir(self.build_lib): return  # this occurs when running setup.py build_ext
        # copy the shared library and executables to the folder where the package is being built
        distutils.file_util.copy_file('Makefile.local', os.path.join(self.build_lib, 'agama'))
        distutils.file_util.copy_file(sharedname, os.path.join(self.build_lib, 'agama'))
        distutils.file_util.copy_file(staticname, os.path.join(self.build_lib, 'agama'))
        distutils.dir_util.copy_tree('exe', os.path.join(self.build_lib, 'agama', 'exe'))
        if os.path.isdir(EXTRAS_DIR):  # this contains third-party libraries built in the process
            distutils.dir_util.copy_tree(EXTRAS_DIR, os.path.join(self.build_lib, 'agama', EXTRAS_DIR), verbose=False)

class MyTest(Command):
    description  = 'run tests'
    user_options = []
    def initialize_options(self): pass
    def finalize_options  (self): pass
    def run(self):
        from py.alltest import alltest
        os.chdir('py')
        alltest()

if '-h' in sys.argv or '--help' in sys.argv: print(helpstr)
forceYes = False
for opt in ['-y','--yes','--assume-yes']:
    if opt in sys.argv:
        forceYes = True
        sys.argv.remove(opt)
if forceYes: say('Assuming yes answers to all interactive questions\n')

if sys.version_info[0]==3 and sys.version_info[1]>=10 and 'install' in sys.argv:
    say('If you are scared by a deprecation warning about running "setup.py install", try "pip install ." instead\n')

distutils.core.setup(
    name             = 'agama',
    version          = '1.0',
    description      = 'Action-based galaxy modelling architecture',
    author           = 'Eugene Vasiliev',
    author_email     = 'eugvas@protonmail.com',
    license          = 'GPL,MIT,BSD',
    url              = 'https://github.com/GalacticDynamics-Oxford/Agama',
    download_url     = 'https://github.com/GalacticDynamics-Oxford/Agama/archive/master.zip',
    long_description = open('README').read(),
    requires         = ['setuptools','wheel','numpy'],
    packages         = ['agama'],
    package_dir      = {'agama': '.'},
    package_data     = {'agama': allFiles('data','doc','py','src','tests') +
        ['Makefile', 'Makefile.msvc', 'Makefile.list', 'Makefile.local.template', 'Doxyfile', 'INSTALL', 'LICENSE', 'NEWS', 'README'] },
    ext_modules      = [distutils.extension.Extension('', [])],
    cmdclass         = {'build_ext': MyBuildExt, 'test': MyTest},
    zip_safe         = False,
    classifiers      = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3']
)
