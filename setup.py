helpstr="""
Installation script for the Agama library.
It could be run just as any regular python program,
> python setup.py install --user
(here --user allows it to be installed without admin privilegies),
or via pip:
> pip install --user .
(here . means that it should take the files from the current folder);
the former approach is more verbose (this could be amended by providing -q flag to setup.py);
the latter one (with pip) might fail on older systems.

The setup script recognizes some environment variables:
CXX  will set the compiler;
CFLAGS  will set additional compilation flags (e.g., a user-defined include path could be provided
by setting CFLAGS=-I/path/to/my/library/include);
LDFLAGS  likewise will set additional link flags (e.g., -L/path/to/my/library/lib -lmylib).

The setup script will attempt to download and compile the required GSL library (if it wasn't found)
and the optional Eigen, CVXOPT and UNSIO libraries
(asking the user for a confirmation, unless command-line arguments include --yes or --assume-yes).
Eigen is header-only so it is only downloaded; UNSIO is downloaded and compiled;
CVXOPT is a python library, so it will be installed by calling "pip install --user cvxopt"
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
import os, sys, platform, subprocess, zipfile, distutils, distutils.core, distutils.dir_util, distutils.file_util
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
            with open('/dev/tty','w') as out:
                out.write(text)
                out.flush()
        except:
            # /dev/tty may not exist or may not be writable!
            pass

# asking a yes/no question and parse the answer (raise an exception in case of ambiguous answer)
def ask(q):
    say(q)
    if forceYes:
        result='y'
        say('y\n')
    else:
        result=sys.stdin.readline()
    if not sys.stdin.isatty() and not result:
        # if the input was redirected and no answer was provided,
        # try to read it directly from the true terminal, if it is available
        try:
            with open('/dev/tty','r') as stdin:
                result=stdin.readline()
            sys.stdout.write(result)  # and duplicate the entered text to stdout, for posterity
        except:
            result=''  # no reply => will produce an error
    result = result.rstrip()
    if not result:
        raise ValueError('No valid response was provided')
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


# find out which required and optional 3rd party libraries are present, and create Makefile.local
def createMakefile():

    say("\n    ==== Checking supported compiler options and available libraries ====\n\n")
    # determine the C++ compiler and any externally provided flags:
    try: CXXFLAGS = os.environ['CFLAGS'].split()
    except KeyError: CXXFLAGS = []
    try: LINK_FLAGS = os.environ['LDFLAGS'].split()
    except KeyError: LINK_FLAGS = []
    # default compilation flags for both the shared library and all example programs that use it
    if MSVC:
        CXXFLAGS += ['/W3', '-wd4244', '-wd4267', '/O2', '/EHsc', '/nologo']
        LINK_DIR_FLAG = '/LIBPATH:'  # prefix for library search path
        COMPILE_OUT_FLAG = '/Fe:'    # prefix for output file
    else:
        CXXFLAGS += ['-fPIC', '-Wall', '-O2']
        LINK_DIR_FLAG = '-L'
        COMPILE_OUT_FLAG = '-o'
    # additional compilation flags for the shared library only (paths to Python, GSL and other third-party libs)
    COMPILE_FLAGS = []
    # additional compilation/linking flags for example programs only (not the shared library)
    EXE_FLAGS = []
    for LINK_FLAG in LINK_FLAGS:   # add paths to libraries to the linking flags for executable programs
        if LINK_FLAG.startswith('-L'): EXE_FLAGS.append(LINK_FLAG)

    # check if a given test code compiles with given flags
    def runCompiler(code='int main(){}\n', flags='', dest=None):
        with open('test.cpp', 'w') as f: f.write(code)
        if dest is None: dest='test.out'
        cmd = '%s %s test.cpp %s %s %s' % (CC, ' '.join(CXXFLAGS+EXE_FLAGS), COMPILE_OUT_FLAG, dest, flags)
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
            success = runCompiler(code=code, flags=flags+' -fPIC -shared', dest=LIB_NAME) \
            and runCompiler(code=EXE_CODE, flags=LIB_NAME, dest=EXE_NAME) \
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

    if os.path.isdir(EXTRAS_DIR+'/include'):   # if it was created in the previous invocation of setup.py
        COMPILE_FLAGS += ['-I'+EXTRAS_DIR+'/include']  # it might already contain some useful stuff
    if os.path.isdir(EXTRAS_DIR+'/lib'):
        LINK_FLAGS += [LINK_DIR_FLAG+EXTRAS_DIR+'/lib']

    # [1a]: check if a compiler exists at all
    if not runCompiler():
        raise CompileError("Could not locate a compiler (set CXX=... environment variable to override)")

    # [1b]: test if OpenMP is supported (optional but highly recommended)
    OMP_FLAG = '-fopenmp' if not MSVC else '/openmp'
    OMP_FLAGS= OMP_FLAG + (' -Werror -Wno-unknown-pragmas' if not MSVC else '')
    OMP_CODE = '#include <omp.h>\n#include <iostream>\nint main(){\nstd::cout << "Number of threads: ";\n'+\
        '#pragma omp parallel\n{std::cout<<\'*\';}\nstd::cout << "\\n";\n}\n'
    if runCompiler(code=OMP_CODE, flags=OMP_FLAGS):
        CXXFLAGS += [OMP_FLAG]
    elif MACOS:
        # on MacOS the clang compiler pretends not to support OpenMP, but in fact it does so
        # if we insist (libomp.so/dylib or libgomp.so must be present in the system for this to work);
        # in some Anaconda installations, though, linking to the system-default libomp.dylib
        # leads to conflicts with libiomp5.dylib, so we first try to link to the latter explicitly.
        CONDA_EXE = os.environ.get('CONDA_EXE')
        if CONDA_EXE is not None and os.path.isfile(CONDA_EXE.replace('bin/conda', 'lib/libiomp5.dylib')) \
            and runCompiler(code=OMP_CODE, flags=CONDA_EXE.replace('bin/conda', 'lib/libiomp5.dylib') +
            ' -Xpreprocessor ' + OMP_FLAGS + ' -I'+CONDA_EXE.replace('bin/conda', 'include/')):
            CXXFLAGS   += ['-Xpreprocessor', OMP_FLAG, '-I'+CONDA_EXE.replace('bin/conda', 'include/')]
            LINK_FLAGS += [CONDA_EXE.replace('bin/conda', 'lib/libiomp5.dylib')]
            EXE_FLAGS  += [CONDA_EXE.replace('bin/conda', 'lib/libiomp5.dylib')]
        elif runCompiler(code=OMP_CODE, flags='-lgomp -Xpreprocessor '+OMP_FLAGS):
            CXXFLAGS   += ['-Xpreprocessor', OMP_FLAG]
            LINK_FLAGS += ['-lgomp']
            EXE_FLAGS  += ['-lgomp']
        elif runCompiler(code=OMP_CODE, flags='-lomp -Xpreprocessor '+OMP_FLAGS):
            CXXFLAGS   += ['-Xpreprocessor', OMP_FLAG]
            LINK_FLAGS += ['-lomp']
            EXE_FLAGS  += ['-lomp']
        elif not ask("Warning, OpenMP is not supported\n"+
            "If you're compiling on MacOS with clang, you'd better install another compiler such as GCC\n"+
            "Do you want to continue without OpenMP? [Y/N] "): exit(1)

    # [1c]: test if C++11 is supported (optional)
    CXX11_FLAG = '-std=c++11'
    if not MSVC and runCompiler(flags=CXX11_FLAG):
        CXXFLAGS += [CXX11_FLAG]

    # [1d]: on MacOS, set the machine architecture explicitly to avoid mismatch between x86_64 and arm64 on the newer M1 platform
    if MACOS:
        ARCH_FLAG = '-arch '+uname[4]  # could be x86_64 or arm64
        if runCompiler(flags=ARCH_FLAG):
            CXXFLAGS += [ARCH_FLAG]

    # [1e]: test the -march flag (optional, allows architecture-dependent compiler optimizations)
    ARCH_FLAG = '-march=native'
    if not MSVC:
        if runCompiler(flags=ARCH_FLAG):
            CXXFLAGS += [ARCH_FLAG]
        else:
            ARCH_FLAG = '-march=core2'  # try a less ambitious option
            if runCompiler(flags=ARCH_FLAG):
                CXXFLAGS += [ARCH_FLAG]

    # [1f]: special treatment for Intel compiler to restore determinism in OpenMP-parallelized loops
    INTEL_FLAG = '-qno-opt-dynamic-align'
    if not MSVC and runCompiler(code='#ifndef __INTEL_COMPILER\n#error\n#endif\nint main(){}\n', flags=INTEL_FLAG):
        CXXFLAGS += [INTEL_FLAG]

    # [1g]: remove a couple of low-importance warnings that are unavoidably generated
    # by the Python C API conventions (only if they are supported by the compiler)
    for WARNING_FLAG in ['-Wno-missing-field-initializers', '-Wno-cast-function-type']:
        if not MSVC and runCompiler(flags='-Werror '+WARNING_FLAG):
            CXXFLAGS += [WARNING_FLAG]

    # [2a]: check that NumPy is present (required by the python interface)
    try:
        import numpy
        NUMPY_INC = '-I"%s"' % numpy.get_include()
    except ImportError:
        raise CompileError("NumPy is not present - python extension cannot be compiled")

    # [2b]: find out the paths to Python.h and libpythonXX.{a,so,dylib,...} (this is rather tricky)
    # and all other relevant compilation/linking flags needed to build a shared library that uses Python
    PYTHON_INC = '-I"%s"' % sysconfig.get_python_inc()

    # various other system libraries that are needed at link time
    PYTHON_LIB_EXTRA = compressList(
        get_config_var('LIBS').split() +
        get_config_var('SYSLIBS').split())

    # test code for a shared library (Python extension module)
    PYTEST_LIB_CODE = """
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
void bla() {PyRun_SimpleString("import sys;print(sys.prefix);");}
void run() {Py_Initialize();bla();Py_Finalize();}
PyMODINIT_FUNC
"""
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
        print("    **** Trying the following options for linking against Python library ****")
        # test code for a program that loads this shared library
        PYTEST_EXE_CODE = 'extern void run();int main(){run();}\n'
        PYTEST_LIB_NAME = './agamatest.so'
        PYTEST_EXE_NAME = './agamatest.exe'
        # try compiling the test shared library
        if not runCompiler(code=PYTEST_LIB_CODE,
            flags=' '.join([PYTHON_INC, NUMPY_INC, '-shared', '-fPIC'] + PYTHON_SO_FLAGS),
            dest=PYTEST_LIB_NAME):
            return False  # the program couldn't be compiled at all (try the next variant)

        # if succeeded, compile the test program that uses this library
        if not runCompiler(code=PYTEST_EXE_CODE,
            flags=' '.join([PYTEST_LIB_NAME] + PYTHON_EXE_FLAGS),
            dest=PYTEST_EXE_NAME) \
            or not os.path.isfile(PYTEST_LIB_NAME) \
            or not os.path.isfile(PYTEST_EXE_NAME):
            return False  # can't find compiled test program
        resultexe = subprocess.Popen(PYTEST_EXE_NAME,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
        # the test program might not be able to find the python home, in which case manually provide it
        if 'Could not find platform independent libraries <prefix>' in resultexe:
            resultexe = subprocess.Popen(PYTEST_EXE_NAME,
                env=dict(os.environ, PYTHONHOME=sys.prefix),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode().rstrip()
        # also try loading this shared library as an extension module
        procpy = subprocess.Popen(sys.executable+" -c 'import agamatest'", shell=True, \
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        resultpy = procpy.communicate()[0].decode().rstrip()
        returnpy = procpy.returncode
        # clean up
        os.remove(PYTEST_EXE_NAME)
        os.remove(PYTEST_LIB_NAME)
        # check if the results (reported library path prefix) are the same as we have in this script
        sysprefix = os.path.realpath(sys.prefix)
        if os.path.realpath(resultexe) != sysprefix or os.path.realpath(resultpy) != sysprefix:
            print("Test program doesn't seem to use the same version of Python, "+\
                "or the library path is reported incorrectly: \n"+\
                "Expected: "+sysprefix+"\n"+\
                "Received: "+resultexe+"\n"+\
                "From py:  "+resultpy+('' if returnpy==0 else ' (crashed with error '+str(returnpy)+')'))
            return False
        print("    **** Successfully linked using these options ****")
        return True   # this combination of options seems reasonable...

    # determine compilation and linking options for the python extension module when using MSVC on Windows
    def findPythonLibMSVC():
        PYTHON_SO_FLAGS = ['/LIBPATH:"%s"' % os.path.join(sys.exec_prefix, 'libs')]
        PYTHON_EXE_FLAGS= []
        if runCompiler(code=PYTEST_LIB_CODE,
            flags=' '.join([PYTHON_INC, NUMPY_INC, '/link', '/dll', '/out:agamatest.pyd'] + PYTHON_SO_FLAGS) ):
            # test the compiled extension module
            procpy = subprocess.Popen('"'+sys.executable+'" -c "import agamatest"', shell=True, \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            resultpy = procpy.communicate()[0].decode().rstrip()
            returnpy = procpy.returncode
            os.remove('agamatest.pyd')   # various cleanups
            tryRemove('agamatest.lib')
            tryRemove('agamatest.exp')
            # check if the results (reported library path prefix) are the same as we have in this script
            sysprefix = os.path.realpath(sys.prefix)
            if returnpy == 0 and os.path.realpath(resultpy) == sysprefix:  # everything fine
                return PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS
            print("Test program doesn't seem to use the same version of Python, "+\
                "or the library path is reported incorrectly: \n"+\
                "Expected: "+sysprefix+"\n"+\
                "Received: "+resultpy+('' if returnpy==0 else ' (crashed with error '+str(returnpy)+')'))
        raise CompileError("Could not compile test program which uses libpython" +
            sysconfig.get_config_var('VERSION'))

    # explore various possible combinations of file name and path to the python library on Linux/MacOS
    def findPythonLib():
        # try linking against the static python library libpython**.a, if this does not succeed,
        # try the shared library libpython**.so** or libpython**.dylib
        PY_VERSION = sysconfig.get_config_var('VERSION')
        for PYTHON_LIB_FILENAME in compressList([sysconfig.get_config_var(x) for x in ['LIBRARY', 'LDLIBRARY', 'INSTSONAME']] +
            ['libpython%s.a' % PY_VERSION, 'libpython%s.so' % PY_VERSION, 'libpython%s.dylib' % PY_VERSION] ):
            for PYTHON_LIB_PATH in compressList([sysconfig.get_config_var(x) for x in ['LIBPL', 'LIBDIR']]):
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
                        print("Trying rpath")
                        RPATH = ['-Wl,-rpath,'+PYTHON_LIB_PATH]  # extend the linker options and try again
                        if tryPythonCode(PYTHON_SO_FLAGS + RPATH):
                            return PYTHON_SO_FLAGS + RPATH, []
                        if "-undefined dynamic_lookup" in sysconfig.get_config_var('LDSHARED'):
                            print("Trying the last resort solution")
                            PYTHON_SO_FLAGS = ['-undefined dynamic_lookup'] + PYTHON_LIB_EXTRA
                            PYTHON_EXE_FLAGS = RPATH + [PYTHON_LIB_FILEPATH]
                            if tryPythonCode(PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS):
                                return PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS

        # if the above efforts did not succeed, try the options suggested by python-config
        PYTHON_CONFIG = 'python%s-config --ldflags' % sysconfig.get_config_var('VERSION')
        print("    **** Trying the options provided by %s ****" % PYTHON_CONFIG)
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
                "The solution is to install a native ARM Python version, e.g. from conda-forge: " +
                "https://docs.conda.io/en/latest/miniconda.html")
        # if none of the above combinations worked, give up...
        raise CompileError("Could not compile test program which uses libpython" +
            sysconfig.get_config_var('VERSION'))

    # [2c]: find the python library and other relevant linking flags
    PYTHON_SO_FLAGS, PYTHON_EXE_FLAGS = findPythonLib() if not MSVC else findPythonLibMSVC()
    COMPILE_FLAGS += ['-DHAVE_PYTHON', PYTHON_INC, NUMPY_INC]
    LINK_FLAGS    += PYTHON_SO_FLAGS
    EXE_FLAGS     += PYTHON_EXE_FLAGS

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
    if not MSVC:   # install GSL on Linux/MacOS
        if runCompileShared(GSL_TEST_CODE, flags=' '.join(COMPILE_FLAGS + LINK_FLAGS + ['-lgsl', '-lgslcblas'])):
            # apparently the headers and libraries can be found in some standard location,
            LINK_FLAGS += ['-lgsl', '-lgslcblas']   # so we only list their names
        else:
            if not ask("GSL library (required) is not found\n"+
                "Should we try to download and compile it now? [Y/N] "): exit(1)
            distutils.dir_util.mkpath(EXTRAS_DIR)
            os.chdir(EXTRAS_DIR)
            say("Downloading GSL\n")
            filename = 'gsl.tar.gz'
            dirname  = 'gsl-2.7'
            try:
                urlretrieve('ftp://ftp.gnu.org/gnu/gsl/gsl-2.7.tar.gz', filename)
                if os.path.isfile(filename):
                    say("Unpacking GSL\n")
                    subprocess.call(['tar', '-zxf', filename])    # unpack the archive
                    os.remove(filename)  # remove the downloaded archive
                    if not os.path.isdir(dirname): raise Exception("Error unpacking GSL")
                else: raise RuntimeError("Cannot find downloaded file")
            except Exception as e:
                raise CompileError(str(e) + "\nError downloading GSL library, aborting...\n"+
                "You may try to manually compile GSL and install it to "+ROOT_DIR+"/"+EXTRAS_DIR+", so that "+
                "the header files are in "+EXTRAS_DIR+"/include and library files - in "+EXTRAS_DIR+"/lib")
            say('Compiling GSL (may take a few minutes)\n')
            result = subprocess.call('(cd '+dirname+'; ./configure --prefix='+os.getcwd()+
                ' CFLAGS="-fPIC -O2" --enable-shared=no; make; make install) > gsl-install.log', shell=True)
            if result != 0 or not os.path.isfile('lib/libgsl.a'):
                 raise CompileError("GSL compilation failed (check "+EXTRAS_DIR+"/gsl-install.log)")
            distutils.dir_util.remove_tree(dirname)  # clean up source and build directories
            COMPILE_FLAGS += ['-I'+EXTRAS_DIR+'/include']
            LINK_FLAGS    += [EXTRAS_DIR+'/lib/libgsl.a', EXTRAS_DIR+'/lib/libgslcblas.a']
            os.chdir(ROOT_DIR)
    else:  # install a GSL port on Windows with MSVC
        if runCompiler(GSL_TEST_CODE + 'int main() {run();}',
            ' '.join(COMPILE_FLAGS + ['/link', 'gsl.lib', 'cblas.lib'] + LINK_FLAGS)):
            LINK_FLAGS += ['gsl.lib', 'cblas.lib']
        else:
            if not ask("GSL library (required) is not found\n"+
                "Should we try to download and compile it now? [Y/N] "): exit(1)
            distutils.dir_util.mkpath(EXTRAS_DIR)
            os.chdir(EXTRAS_DIR)
            say("Downloading GSL\n")
            filename = 'gsl.zip'
            dirname  = 'gsl-532847499424f4c04ef44ec4801f3219d223ede2'
            try:
                urlretrieve('https://github.com/BrianGladman/gsl/archive/532847499424f4c04ef44ec4801f3219d223ede2.zip', filename)
                if os.path.isfile(filename):
                    say("Unpacking GSL\n")
                    zipf = zipfile.ZipFile(filename, 'r')  # unpack the archive
                    zipf.extractall()
                    zipf.close()
                    os.remove(filename)  # remove the downloaded archive
                    if not os.path.isdir(dirname): raise Exception("Error unpacking GSL")
                else: raise RuntimeError("Cannot find downloaded file")
            except Exception as e:
                raise CompileError(str(e) + "\nError downloading GSL library, aborting...\n"+
                "You may try to manually compile GSL and install it to "+ROOT_DIR+"/"+EXTRAS_DIR+", so that "+
                "the header files are in "+EXTRAS_DIR+"/include and library files - in "+EXTRAS_DIR+"/lib")
            say('Compiling GSL (may take a few minutes)\n')
            os.chdir(dirname+'\\build.vc')
            result = subprocess.call('msbuild /v:n gslhdrs  /p:Configuration=Release >>../../gsl-install.log', shell=True)
            result+= subprocess.call('msbuild /v:n cblaslib /p:Configuration=Release >>../../gsl-install.log', shell=True)
            result+= subprocess.call('msbuild /v:n gsllib   /p:Configuration=Release >>../../gsl-install.log', shell=True)
            if result != 0 or not os.path.isfile('lib/x64/Release/gsl.lib'):
                raise CompileError("GSL compilation failed (check "+EXTRAS_DIR+"/gsl-install.log)")
            # copy the static libraries to extras/lib
            distutils.dir_util.mkpath('../../lib')
            distutils.file_util.copy_file('lib/x64/Release/gsl.lib',   '../../lib/')
            distutils.file_util.copy_file('lib/x64/Release/cblas.lib', '../../lib/')
            distutils.dir_util.copy_tree('../gsl', '../../include/gsl', verbose=False)
            os.chdir('../..')
            # delete the compiled directory
            distutils.dir_util.remove_tree(dirname)  # clean up source and build directories
            COMPILE_FLAGS += ['-I'+EXTRAS_DIR+'/include']
            LINK_FLAGS += [LINK_DIR_FLAG+EXTRAS_DIR+'/lib', 'gsl.lib', 'cblas.lib']
            os.chdir(ROOT_DIR)
            if not runCompiler(GSL_TEST_CODE + 'int main() {run();}',
                ' '.join(COMPILE_FLAGS + ['/link'] + LINK_FLAGS)):
                    raise CompileError("GSL compilation did not succeed")

    # [4]: test if Eigen library is present (optional)
    if runCompiler(code='#include <Eigen/Core>\nint main(){}\n', flags=' '.join(COMPILE_FLAGS)):
        COMPILE_FLAGS += ['-DHAVE_EIGEN']
    else:
        if ask("Eigen library (recommended) is not found\n"+
            "Should we try to download it now (no compilation needed)? [Y/N] "):
            distutils.dir_util.mkpath(EXTRAS_DIR+'/include/unsupported')
            os.chdir(EXTRAS_DIR)
            say('Downloading Eigen\n')
            filename = 'Eigen.zip'
            dirname  = 'eigen-git-mirror-3.3.7'
            try:
                urlretrieve('https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.zip', filename)
                if os.path.isfile(filename):
                    say("Unpacking Eigen\n")
                    zipf = zipfile.ZipFile(filename, 'r')
                    zipf.extractall()
                    zipf.close()
                    if os.path.isdir(dirname):
                        distutils.dir_util.copy_tree(dirname+'/Eigen', 'include/Eigen', verbose=False)  # copy the headers
                        distutils.dir_util.copy_tree(dirname+'/unsupported/Eigen', 'include/unsupported/Eigen', verbose=False)
                        distutils.dir_util.remove_tree(dirname)  # and delete the rest
                        COMPILE_FLAGS += ['-DHAVE_EIGEN', '-I'+EXTRAS_DIR+'/include']
                    os.remove(filename)                          # remove the downloaded archive
                else: raise RuntimeError("Cannot find downloaded file")
            except Exception as e:
                say("Failed to install Eigen: "+str(e)+"\n")     # didn't succeed with Eigen
            os.chdir(ROOT_DIR)

    # [5a]: test if CVXOPT is present (optional); install if needed
    try:
        import cvxopt  # import the python module
    except:  # import error or some other problem, might be corrected
        if ask("CVXOPT library (needed only for Schwarzschild modelling) is not found\n"
            "Should we try to install it now? [Y/N] "):
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'cvxopt'])
            except Exception as e:
                say("Failed to install CVXOPT: "+str(e)+"\n")

    # [5b]: if the cvxopt module is available in Python, make sure that we also have C header files
    try:
        import cvxopt   # if this fails, skip cvxopt altogether
        if runCompiler(code='#include <cvxopt.h>\nint main(){import_cvxopt();}\n',
            flags=' '.join(['-c', PYTHON_INC, NUMPY_INC] + COMPILE_FLAGS)):
            COMPILE_FLAGS += ['-DHAVE_CVXOPT']
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
                COMPILE_FLAGS += ['-DHAVE_CVXOPT', '-I'+EXTRAS_DIR+'/include']
            else:
                say("Failed to download CVXOPT header files, this feature will not be available\n")
    except: pass  # cvxopt wasn't available

    # [6]: test if GLPK is present (optional - ignored if not found)
    if not MSVC and runCompileShared('#include <glpk.h>\nvoid run() { glp_create_prob(); }\n',
        flags=' '.join(COMPILE_FLAGS + LINK_FLAGS + ['-lglpk'])):
        COMPILE_FLAGS += ['-DHAVE_GLPK']
        LINK_FLAGS    += ['-lglpk']
    else:
        say("GLPK library (optional) is not found, ignored\n")

    # [7]: test if UNSIO is present (optional), download and compile if needed
    if not MSVC and runCompileShared('#include <uns.h>\nvoid run() { }\n',
        flags=' '.join(COMPILE_FLAGS + LINK_FLAGS + ['-lunsio', '-lnemo'])):
        COMPILE_FLAGS += ['-DHAVE_UNSIO']
        LINK_FLAGS    += ['-lunsio', '-lnemo']
    elif not MSVC:
        if ask("UNSIO library (optional; used for input/output of N-body snapshots) is not found\n"+
            "Should we try to download and compile it now? [Y/N] "):
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
                    say("Compiling UNSIO\n")
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
                        if runCompiler(code='#include <uns.h>\nint main(){}\n', flags=' '.join(UNSIO_COMPILE_FLAGS+UNSIO_LINK_FLAGS)):
                            COMPILE_FLAGS += ['-DHAVE_UNSIO'] + UNSIO_COMPILE_FLAGS
                            LINK_FLAGS    += UNSIO_LINK_FLAGS
                        else:
                            raise CompileError('Failed to link against the just compiled UNSIO library')
                    else: raise CompileError(
                        "Failed compiling UNSIO (check "+EXTRAS_DIR+"/unsio-install.log)")
            except Exception as e:  # didn't succeed with UNSIO
                say(str(e)+'\n')

    # [99]: put everything together and create Makefile.local
    with open('Makefile.local','w') as f: f.write(
        ("# set the default compiler if no value is found in the environment variables or among command-line arguments\n" +
        "ifeq ($(origin CXX),default)\nCXX = " + CC + "\nendif\n" +
        "ifeq ($(origin FC), default)\nFC  = gfortran\nendif\nLINK = $(CXX)\n" if not MSVC else "") +
        "# compilation/linking flags for both the shared library and any programs that use it\n" +
        "CXXFLAGS      = " + " ".join(compressList(CXXFLAGS)) + "\n" +
        "# compilation flags for the shared library only (files in src/)\n" +
        "COMPILE_FLAGS = " + " ".join(compressList(COMPILE_FLAGS)) + "\n" +
        "# linking flags for the shared library only\n" +
        "LINK_FLAGS    = " + " ".join(compressList(LINK_FLAGS)) + "\n" +
        ("# linking flags for the example/test programs\n" +
        "EXE_FLAGS     = " + " ".join(compressList(EXE_FLAGS)) + "\n" if EXE_FLAGS else "") )


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
            make   = 'nmake -f Makefile.msvc'
            soname = 'agama.pyd'
        else:  # standard unix (including macos)
            make   = 'make'
            soname = 'agama.so'
        if subprocess.call(make) != 0 or not os.path.isfile(soname):
            raise CompileError("Compilation failed")
        if not os.path.isdir(self.build_lib): return  # this occurs when running setup.py build_ext
        # copy the shared library and executables to the folder where the package is being built
        distutils.file_util.copy_file('Makefile.local', os.path.join(self.build_lib, 'agama'))
        distutils.file_util.copy_file(soname, os.path.join(self.build_lib, 'agama'))
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
    requires         = ['numpy'],
    packages         = ['agama'],
    package_dir      = {'agama': '.'},
    package_data     = {'agama': allFiles('data','doc','py','src','tests') +
        ['Makefile', 'Makefile.local.template', 'Doxyfile', 'INSTALL', 'LICENSE', 'README'] },
    ext_modules      = [distutils.extension.Extension('', [])],
    cmdclass         = {'build_ext': MyBuildExt, 'test': MyTest},
    zip_safe         = False,
    classifiers      = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3']
)
