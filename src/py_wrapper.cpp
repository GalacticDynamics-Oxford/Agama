#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define DEBUGPRINT
#include <numpy/arrayobject.h>
#include "utils_config.h"
#include "potential_factory.h"
#include "math_spline.h"

/// \name  Some general definitions and utility functions
///@{

/// arguments of functions
struct ArgDescription {
    const char* name;  ///< argument name
    char type;         ///< letter that determines argument type ('s','d','i','O')
    const char* descr; ///< textual description
};

/// argument type in human-readable words
static const char* argTypeDescr(char type)
{
    switch(type) {
        case 's': return "string";
        case 'i': return "int";
        case 'd': return "float";
        case 'O': return "object";
        default:  return "unknown type";
    }
}

/// max number of keywords
static const size_t MAX_NUM_KEYWORDS = 64;

/// max size of docstring
static const size_t MAX_LEN_DOCSTRING = 4096;

/// return size of array-like type
static int listOrArraySize(PyObject* a)
{
    if(a && PyList_Check(a))
        return static_cast<int>(PyList_Size(a));
    if(a && PyArray_Check((PyArrayObject *)a) &&
        PyArray_ISFLOAT(const_cast<const PyArrayObject*>((PyArrayObject*)a)) &&
        PyArray_NDIM(const_cast<const PyArrayObject*>((PyArrayObject*)a))==1 )
        return static_cast<int>(*PyArray_DIMS((PyArrayObject*)a));
    return -1;
}

static double listOrArrayElem(PyObject* a, int index)
{
    if(PyList_Check(a))
        return PyFloat_AsDouble(PyList_GetItem(a, index));
    else {
        npy_intp i=static_cast<npy_intp>(index);
        void* data=PyArray_GetPtr((PyArrayObject*)a, &i);
        if(PyArray_TYPE((PyArrayObject*)a)==NPY_FLOAT) return *((float*)data);
        if(PyArray_TYPE((PyArrayObject*)a)==NPY_DOUBLE) return *((double*)data);
        PyErr_SetString(PyExc_ValueError, "Unknown data type in array");
        return 0;
    }
}

///@}
/// \name   Potential class and related data
///@{

/// Python type corresponding to Potential class
typedef struct {
    PyObject_HEAD
    const potential::BasePotential* pot;
} PotentialObject;

/*static PyObject* PotentialObject_new(PyTypeObject *type, PyObject*, PyObject*)
{
    PotentialObject *self = (PotentialObject*)type->tp_alloc(type, 0);
    if(self) self->pot=NULL;
    printf("Allocated a new Potential object\n");
    return (PyObject*)self;
}*/

static void Potential_dealloc(PotentialObject* self)
{
    if(self->pot) {
#ifdef DEBUGPRINT
        printf("Deleted an instance of %s potential\n", self->pot->name());
#endif
        delete self->pot;
    }
#ifdef DEBUGPRINT
    else printf("Deleted empty Potential\n");
#endif
    self->ob_type->tp_free((PyObject*)self);
}

/// list of all possible arguments of Potential class constructor
static const ArgDescription potentialArgs[] = {
    {"file",              's', "the name of ini file, potential coefficients file, or N-body snapshot file"},
    {"type",              's', "potential type, such as 'Plummer', 'Ferrers', or potential expansion type, such as 'SplineExp'"},
    {"density",           's', "density model for potential expansion, like 'Dehnen', 'MGE'"},
    {"symmetry",          's', "assumed symmetry for potential expansion constructed from an N-body snapshot"},
    {"points",            'O', "array of point masses to be used in construction of a potential expansion"},
    {"mass",              'd', "total mass of the model"},
    {"scaleradius",       'd', "scale radius of the model (if applicable)"},
    {"scaleradius2",      'd', "second scale radius of the model (if applicable)"},
    {"q",                 'd', "axis ratio y/x, i.e., intermediate to long axis (if applicable)"},
    {"p",                 'd', "axis ratio z/x, i.e., short to long axis (if applicable)"},
    {"gamma",             'd', "central cusp slope (applicable for Dehnen model)"},
    {"sersicindex",       'd', "Sersic index (applicable for Sersic density model)"},
    {"numcoefsradial",    'i', "number of radial terms in BasisSetExp or grid points in spline potentials"},
    {"numcoefsangular",   'i', "order of spherical-harmonic expansion (max.index of angular harmonic coefficient)"},
    {"numcoefsvertical",  'i', "number of coefficients in z-direction for CylSplineExp potential"},
    {"alpha",             'd', "parameter that determines functional form of BasisSetExp potential"},
    {"splinesmoothfactor",'d', "amount of smoothing in SplineExp initialized from an N-body snapshot"},
    {"splinermin",        'd', "if nonzero, specifies the innermost grid node radius for SplineExp and CylSplineExp"},
    {"splinermax",        'd', "if nonzero, specifies the outermost grid node radius for SplineExp and CylSplineExp"},
    {"splinezmin",        'd', "if nonzero, specifies the z-value of the innermost grid node in CylSplineExp"},
    {"splinezmax",        'd', "if nonzero, specifies the z-value of the outermost grid node in CylSplineExp"},
    {NULL,0,NULL}
};

/// this string will contain full list of parameters and other relevant info to be printed out via `help(Potential)`
static char docstringPotential[MAX_LEN_DOCSTRING];

static const char* keywordsPotential[MAX_NUM_KEYWORDS] = {NULL};

static char keywordTypesPotential[MAX_NUM_KEYWORDS+1] = "|";

/// build the docstring for Potential class
static void buildDocstringPotential()
{
    // header line, store the number of symbols printed
    unsigned int bufIndex = snprintf(docstringPotential, MAX_LEN_DOCSTRING,
        "Potential is a class that represents a wide range of gravitational potentials\n"
        "There are a number of possible named arguments for the constructor:\n\n");
    unsigned int argIndex = 0;
    while(potentialArgs[argIndex].name != NULL && bufIndex < MAX_LEN_DOCSTRING && argIndex < MAX_NUM_KEYWORDS) {
        bufIndex += snprintf(docstringPotential + bufIndex, MAX_LEN_DOCSTRING - bufIndex,
            "    %s (%s) - %s\n", potentialArgs[argIndex].name,
            argTypeDescr(potentialArgs[argIndex].type), potentialArgs[argIndex].descr);
        keywordsPotential[argIndex] = potentialArgs[argIndex].name;
        keywordTypesPotential[argIndex+1] = potentialArgs[argIndex].type;
        argIndex++;
    }
    if(bufIndex < MAX_LEN_DOCSTRING) {
        bufIndex += snprintf(docstringPotential + bufIndex, MAX_LEN_DOCSTRING - bufIndex,
            "\nRequired parameters are either 'type' or 'file' (or both)\n");
    }
    if(bufIndex >= MAX_LEN_DOCSTRING || argIndex >= MAX_NUM_KEYWORDS)
    {   // overflow shouldn't occur, but if it does, issue a warning
        printf("WARNING: Could not properly initialize Potential class docstring\n");
    }
}

static int Potential_init(PyObject* self, PyObject* args, PyObject* namedargs)
{
    potential::ConfigPotential cfg;
    const char* file="";
    const char* type="";
    const char* density="";
    const char* symmetry="";
    PyObject* points=NULL;

    // it is VITALLY IMPORTANT to list all data fields in exactly the same order as appeared in the `potentialArgs` array!!!
    if(!PyArg_ParseTupleAndKeywords(args, namedargs, keywordTypesPotential, const_cast<char**>(keywordsPotential),
        &file, &type, &density, &symmetry,
        &points,
        &(cfg.mass), &(cfg.scaleRadius), &(cfg.scaleRadius2), &(cfg.q), &(cfg.p), &(cfg.gamma), &(cfg.sersicIndex),
        &(cfg.numCoefsRadial), &(cfg.numCoefsAngular), &(cfg.numCoefsVertical), 
        &(cfg.alpha), &(cfg.splineSmoothFactor), &(cfg.splineRMin), &(cfg.splineRMax), &(cfg.splineZMin), &(cfg.splineZMax) ))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to the Potential constructor;\n"
            "type 'help(Potential)' to get the list of possible arguments and their types");
        return -1;
    }
#ifdef DEBUGPRINT
    utils::KeyValueMap kv;
    storeConfigPotential(cfg, kv);
    printf("%s", kv.dump().c_str());
#endif
    cfg.fileName = std::string(file);
    cfg.potentialType = potential::getPotentialTypeByName(type);
    cfg.densityType = potential::getDensityTypeByName(density);
    cfg.symmetryType = potential::getSymmetryTypeByName(symmetry);
    if(cfg.potentialType == potential::PT_UNKNOWN) {
        if(type[0]==0)
            PyErr_SetString(PyExc_ValueError, "Should provide type='...' or file='...' parameter");
        else
            PyErr_SetString(PyExc_ValueError, "Incorrect type='...' parameter");
        return -1;
    }
    try{  // the code below may generate exceptions which shouldn't propagate to Python
        const potential::BasePotential* pot=NULL;
        if(points!=NULL) {  // a list of particles was provided
            if( file[0]!=0 ) {
                PyErr_SetString(PyExc_ValueError, "Cannot provide both points and filename");
                return -1;
            }
            if( cfg.potentialType != potential::PT_BSE &&
                cfg.potentialType != potential::PT_SPLINE &&
                cfg.potentialType != potential::PT_CYLSPLINE ) {
                PyErr_SetString(PyExc_ValueError, "Potential should be of an expansion type");
                return -1;
            }
            // check type
            int numpt = listOrArraySize(points);
            if(numpt<=0 || numpt%4!=0) {
                PyErr_SetString(PyExc_ValueError, "Empty array of points");
                return -1;
            }
            numpt/=4;
            particles::PointMassArray<coord::PosCar> pointArray;
            pointArray.data.reserve(numpt);
            for(int i=0; i<numpt; i++) {
                double x=listOrArrayElem(points, i);
                double y=listOrArrayElem(points, i+numpt);
                double z=listOrArrayElem(points, i+numpt*2);
                double m=listOrArrayElem(points, i+numpt*3);
                pointArray.add(coord::PosCar(x,y,z), m);
            }
            if(PyErr_Occurred()) {  // numerical conversion error
                return -1;
            }
            pot = potential::createPotentialFromPoints(cfg, pointArray);
        } else {
            pot = potential::createPotential(cfg);  // attempt to create potential from configuration parameters
        }
        if(!pot) {  // shouldn't get here -- either everything is ok, or we got a C++ exception already
            PyErr_SetString(PyExc_ValueError, "Incorrect parameters for the potential");
            return -1;
        }
#ifdef DEBUGPRINT
        printf("Created an instance of %s potential\n", pot->name());
#endif
        if(((PotentialObject*)self)->pot) {
#ifdef DEBUGPRINT
            printf("Deleted previous instance of %s potential\n", ((PotentialObject*)self)->pot->name());
#endif
            delete ((PotentialObject*)self)->pot;
        }
        ((PotentialObject*)self)->pot = pot;
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in creating potential: ")+e.what()).c_str());
        return -1;
    }
}

static bool Potential_isCorrect(PyObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_ValueError, "Should be called as method of Potential object");
        return false;
    }
    if(((PotentialObject*)self)->pot==NULL) {
        PyErr_SetString(PyExc_ValueError, "Potential is not initialized properly");
        return false;
    }
    return true;
}

static PyObject* Potential_potential(PyObject* self, PyObject* args)
{
    coord::PosCar pos;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "ddd", &pos.x, &pos.y, &pos.z))
        return NULL;
    return Py_BuildValue("d", ((PotentialObject*)self)->pot->value(pos));
}

static PyObject* Potential_density(PyObject* self, PyObject* args)
{
    coord::PosCar pos;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "ddd", &pos.x, &pos.y, &pos.z))
        return NULL;
    return Py_BuildValue("d", ((PotentialObject*)self)->pot->density(pos));
}

static PyObject* Potential_force(PyObject* self, PyObject* args)
{
    coord::PosCar pos;
    coord::GradCar grad;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "ddd", &pos.x, &pos.y, &pos.z))
        return NULL;
    ((PotentialObject*)self)->pot->eval(pos, 0, &grad);
    return Py_BuildValue("ddd", -grad.dx, -grad.dy, -grad.dz);
}

static PyObject* Potential_force_deriv(PyObject* self, PyObject* args)
{
    coord::PosCar pos;
    coord::GradCar grad;
    coord::HessCar hess;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "ddd", &pos.x, &pos.y, &pos.z))
        return NULL;
    ((PotentialObject*)self)->pot->eval(pos, 0, &grad, &hess);
    return Py_BuildValue("(ddd)(dddddd)", -grad.dx, -grad.dy, -grad.dz,
         -hess.dx2, -hess.dy2, -hess.dz2, -hess.dxdy, -hess.dydz, -hess.dxdz);
}

static PyObject* Potential_name(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return Py_BuildValue("s", ((PotentialObject*)self)->pot->name());
}

static PyObject* Potential_export(PyObject* self, PyObject* args)
{
    const char* filename=NULL;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    potential::PotentialType t = potential::getPotentialType(*((PotentialObject*)self)->pot);
    if (t != potential::PT_BSE &&
        t != potential::PT_SPLINE &&
        t != potential::PT_CYLSPLINE ) {
        PyErr_SetString(PyExc_ValueError, "Potential is not of an expansion type");
        return NULL;
    }
    try{
        writePotential(filename, *((PotentialObject*)self)->pot);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error writing file: ")+e.what()).c_str());
        return NULL;
    }
}

static PyMethodDef Potential_methods[] = {
    { "name", (PyCFunction)Potential_name, METH_NOARGS, 
      "Return the name of the potential\nNo arguments\nReturns: string" },
    { "potential", Potential_potential, METH_VARARGS, 
      "Compute potential at given point\nArguments: x,y,z (float)\nReturns: float" },
    { "density", Potential_density, METH_VARARGS, 
      "Compute density at given point\nArguments: x,y,z (float)\nReturns: float" },
    { "force", Potential_force, METH_VARARGS, 
      "Compute force at given point\nArguments: x,y,z (float)\nReturns: float[3] - x,y,z components of force" },
    { "force_deriv", Potential_force_deriv, METH_VARARGS, 
      "Compute force and its derivatives at given point\nArguments: x,y,z (float)\nReturns: (float[3],float[6]) - "
      "x,y,z components of force, and the matrix of force derivatives stored as dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx" },
    { "export", (PyCFunction)Potential_export, METH_VARARGS, 
      "Export potential expansion coefficients to a text file\nArguments: filename (string)\nReturns: none" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0, "py_wrapper.Potential",
    sizeof(PotentialObject), 0, (destructor)Potential_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Potential_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringPotential, 
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, 0, 0, 0, 0, 0,
    Potential_init, 0, /*Potential_new*/
};

///@}
/// \name  SplineApprox class
///@{

typedef struct {
    PyObject_HEAD
    math::CubicSpline* spl;
} SplineApproxObject;

static void SplineApprox_dealloc(SplineApproxObject* self)
{
    if(self->spl)
        delete self->spl;
    self->ob_type->tp_free((PyObject*)self);
}

const char* class_docstring_spline = 
    "SplineApprox is a class that deals with smoothing splines\n";

static int SplineApprox_init(PyObject* self, PyObject* args, PyObject* namedargs)
{
    ((SplineApproxObject*)self)->spl=NULL;
    static const char* keywords[] = {"x","y","k","s",NULL};
    PyObject* ptx=NULL;
    PyObject* pty=NULL;
    PyObject* ptk=NULL;
    double smoothfactor=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedargs, "OOO|d", const_cast<char **>(keywords),
        &ptx, &pty, &ptk, &smoothfactor)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters passed to the SplineApprox constructor");
        return -1;
    }
    int numx=listOrArraySize(ptx);
    int numy=listOrArraySize(pty);
    int numk=listOrArraySize(ptk);
    if(numx<=0 || numy<=0 || numx!=numy || numk<4) {
        PyErr_SetString(PyExc_ValueError, 
        "Arguments must be two arrays of equal length (x and y) and a third array (knots, at least 4), "
        "and optionally a float (smooth factor)");
        return -1;
    }
    std::vector<double> xvalues(numx), yvalues(numx), knots(numk), splinevals;
    double der1, der2;  // endpoint derivatives
    for(int i=0; i<numx; i++) {
        xvalues[i]=listOrArrayElem(ptx, i);
        yvalues[i]=listOrArrayElem(pty, i);
    }
    for(int i=0; i<numk; i++) 
        knots[i]=listOrArrayElem(ptk, i);
    if(PyErr_Occurred())  // numerical conversion error
        return -1;
    try{
        math::SplineApprox spl(xvalues, knots);
        if(smoothfactor>0)
            spl.fitDataOversmooth(yvalues, smoothfactor, splinevals, der1, der2);
        else
            spl.fitData(yvalues, -smoothfactor, splinevals, der1, der2);
        ((SplineApproxObject*)self)->spl = new math::CubicSpline(knots, splinevals, der1, der2);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in spline initialization: ")+e.what()).c_str());
        return -1;
    }
    return 0;
}

double spl_eval(const math::CubicSpline* spl, double x, int der=0)
{
    double result;
    switch(der) {
        case 0: return spl->value(x);
        case 1: spl->evalDeriv(x, NULL, &result); return result;
        case 2: spl->evalDeriv(x, NULL, NULL, &result); return result;
        default: return NAN;
    }
}

static PyObject* SplineApprox_value(PyObject* self, PyObject* args, PyObject* /*kw*/)
{
    PyObject* ptx=NULL;
    int der=0;
    if(self==NULL || ((SplineApproxObject*)self)->spl==NULL || !PyArg_ParseTuple(args, "O|i", &ptx, &der))
        return NULL;
    if(der>2) {
        PyErr_SetString(PyExc_ValueError, "Can only compute derivatives up to 2nd");
        return NULL;
    }
    if(PyFloat_Check(ptx))  // one value
        return Py_BuildValue("d", spl_eval(((SplineApproxObject*)self)->spl, PyFloat_AsDouble(ptx), der) );
    // else an array of values
    int num=listOrArraySize(ptx);
    if(num<=0) {
        PyErr_SetString(PyExc_ValueError, "Argument must be either float, list or numpy array");
        return NULL;
    }
    npy_intp dims[1]={num};
    PyArrayObject* ptv = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    for(int i=0; i<num; i++)
        ((double*)PyArray_DATA(ptv))[i] = spl_eval(((SplineApproxObject*)self)->spl, listOrArrayElem(ptx, i), der);
    return PyArray_Return(ptv);
}

static PyMethodDef SplineApprox_methods[] = {
    { NULL, NULL, 0, NULL }  // no named methods
};

static PyTypeObject SplineApproxType = {
    PyObject_HEAD_INIT(NULL)
    0, "py_wrapper.SplineApprox",
    sizeof(SplineApproxObject), 0, (destructor)SplineApprox_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, SplineApprox_value, /*SplineApprox_status*/0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, class_docstring_spline, 
    0, 0, 0, 0, 0, 0, SplineApprox_methods, 0, 0, 0, 0, 0, 0, 0,
    SplineApprox_init, 0, 
};
///@}

static PyMethodDef py_wrapper_methods[] = {
    {NULL}
};

PyMODINIT_FUNC
initpy_wrapper(void)
{
    PyObject* mod = Py_InitModule("py_wrapper", py_wrapper_methods);
    if(!mod) return;

    // Potential class
    buildDocstringPotential();
    PotentialType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PotentialType) < 0) return;
    Py_INCREF(&PotentialType);
    PyModule_AddObject(mod, "Potential", (PyObject *)&PotentialType);

    SplineApproxType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&SplineApproxType) < 0) return;
    Py_INCREF(&SplineApproxType);
    PyModule_AddObject(mod, "SplineApprox", (PyObject *)&SplineApproxType);

    import_array();
}

/// \endcond