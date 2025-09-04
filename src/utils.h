/** \file    utils.h
    \brief   several string formatting and error reporting routines
    \author  EV
    \date    2009-2016
*/
#pragma once
#include <string>
#include <vector>

/** Helper routines for string handling, logging and miscellaneous other tasks.  */
namespace utils {

/* ------- logging routines ------- */

// the following definitions ensure that function name will appear
// with full class qualifier in calls to "msg"
#ifdef __GNUC__
#define FUNCNAME __PRETTY_FUNCTION__
#else
#define FUNCNAME __FUNCTION__
#endif

/// level of verbosity for logging messages
enum VerbosityLevel {
    VL_DISABLE = -1,  ///< same as the default setting, but suppress the progress indicator in Python
    VL_MESSAGE = 0,   ///< information messages that are always printed (default)
    VL_WARNING = 1,   ///< important but non-critical warnings indicating possible problems
    VL_DEBUG   = 2,   ///< ordinary debugging messages
    VL_VERBOSE = 3,   ///< copious amount of debugging messages and various data written to text files
};

/** Prototype of a function that displays a message.
    The default routine prints the message if its level does not exceed the global 
    verbosityLevel variable (initialized from the environment variable LOGLEVEL);
    the text is sent to stderr or to a log file, depending on the environment variable LOGFILE.
    \param[in] level is the verbosity level of the message;
    \param[in] origin is the name of calling function (should contain the value of FUNCNAME macro);
    \param[in] message is the text to be displayed.
*/
typedef void MsgType(VerbosityLevel level, const char* origin, const std::string &message);

/** global pointer to the logging routine;
    by default it prints messages to stderr or to a file defined by the environment variable LOGFILE,
    but it may be replaced with any user-defined routine.
*/
extern MsgType* msg;

/** global settings for the verbosity level of information or debugging messages;
    initialized at startup from the environment variable LOGLEVEL
    (if it exists, otherwise set to default VL_MESSAGE).
*/
extern VerbosityLevel verbosityLevel;

/** a hacky way of implementing "lazy logging": if the LEVEL argument is above verbosityLevel,
    not only the message is discarded, but even the other arguments (ORIGIN, MESSAGE)
    are not evaluated, saving effort of assembling a log message that is never used.
    The weird construction "do {...} while(false)" enables the usage of this macro as if it were
    a proper function call, i.e. with a trailing semicolon (also in conditional statements).
*/
#define FILTERMSG(LEVEL, ORIGIN, MESSAGE) \
    do{ if(LEVEL <= utils::verbosityLevel) { utils::msg(LEVEL, ORIGIN, MESSAGE); } } while(false)


/** return a textual representation of the stack trace */
std::string stacktrace();


/** Helper class for monitoring the Control-Break signal.
    In a computationally heavy section of code, one may set up a custom signal handler
    and periodically check if it was triggered, terminating the computation early if requested.
    The C++ routine could look like this:
    ~~~~
    void busyLoop() {
        utils::CtrlBreakHandler cbrk;
        for(int i=0; i<1000000000; i++) {
            if(cbrk.triggered())
                throw std::runtime_error("Keyboard interrupt");
        }
    }
    ~~~~
    The signal handler is restored automatically, whether the routine exits normally or via
    an exception. The calling code then could check the exception and decide what to do.
    In a pure C++ program, an uncaught exception would usually terminate it anyway,
    as the Control-Break signal is intended to do.
    In the Python interface, though, we translate the C++ exception into an equivalent Python
    exception, which then will be dealt within the script (or simply ignored in an interactive
    session).

    It is safe to instantiate this class multiple times, in nested routines.
*/
class CtrlBreakHandler {
public:
    CtrlBreakHandler();      ///< sets up a custom signal handler
    ~CtrlBreakHandler();     ///< restores the previous signal handler
    static bool triggered(); ///< returns true if the Ctrl-Break signal was received, false otherwise
    static std::string message() { return "Keyboard interrupt"; }  ///< standardized exception message
};


/*------------- string functions ------------*/

/** split a string into several items.
    \param[in]  src -- string to be split;
    \param[in]  delim -- array of characters used as delimiters;
    \return  an array of non-empty strings
    (if string contained nothing but delimiters, the returned array has zero length).
*/
std::vector<std::string> splitString(const std::string& src, const std::string& delim);

/// check if a string ends with another string
bool endsWithStr(const std::string& str, const std::string& end);

/// compare two strings in a case-insensitive way
bool stringsEqual(const std::string& str1, const std::string& str2);

/// overloaded routine that compares two strings in a case-insensitive way
bool stringsEqual(const std::string& str1, const char* str2);


/** convert a string to a number: initial whitespace is skipped,
    any non-parseable characters after the number are silently ignored;
    \throw std::invalid_argument exception if there is no valid number at all
    (however, if the string is empty, 0 is returned without triggering an exception)
*/
int toInt(const char* val);
float toFloat(const char* val);
double toDouble(const char* val);

/** convert a string to a boolean value: skip any initial and final whitespace, and
    return true if the string starts with 'y', 't' or '1' (case-insensitive),
    false if the string starts with 'n', 'f', '0' or is empty;
    \throw std::invalid_argument exception if neither of these conditions holds
*/
bool toBool(const char* val);

// handy overloads of the above functions that take std::string, not const char*, as input
inline int toInt(const std::string& val)
{ return toInt(val.c_str()); }
inline float toFloat(const std::string& val)
{ return toFloat(val.c_str()); }
inline double toDouble(const std::string& val)
{ return toDouble(val.c_str()); }
inline bool toBool(const std::string& val)
{ return toBool(val.c_str()); }

/// convert an array of strings (possibly created by splitString) to an array of doubles
inline std::vector<double> toDoubleVector(const std::vector<std::string>& stringVector)
{
    std::vector<double> doubleVector(stringVector.size());
    for(unsigned int i=0; i<stringVector.size(); i++)
        doubleVector[i] = toDouble(stringVector[i]);
    return doubleVector;
}

/// convert a bool to a string
inline static std::string toString(bool val)
{ return val?"true":"false"; }

/// convert a number to a string with a given precision (number of significant digits)
std::string toString(double val, unsigned int width=6);
std::string toString(float val, unsigned int width=6);
std::string toString(int val);
std::string toString(unsigned int val);
std::string toString(long val);
std::string toString(unsigned long val);
std::string toString(long long val);
std::string toString(unsigned long long val);

/// convert a pointer to a string
std::string toString(const void* val);
/// convert any typed pointer to a string
template<typename T>
inline std::string toString(const T* val)
{
    return toString(static_cast<const void*>(val));
}

/// convert a vector to a string with values separated by the given character
template<typename T>
std::string toString(const std::vector<T>& vec, char separator=',')
{
    std::string result;
    for(size_t i=0; i<vec.size(); i++) {
        if(i>0) result += separator;
        result += toString(vec[i]);
    }
    return result;
}

/// Pretty-print: convert floating-point or integer numbers to a string of exactly fixed length.
/// Choose fixed-point or exponential format depending on which one is more accurate;
/// if the number does not fit into the given width, return a string with # symbols.
std::string pp(double num, unsigned int width);

/// check if the character starts a comment
inline bool isComment(const char c) { return c=='#' || c==';'; }


/*------- time measurement -------*/

/// measurement of elapsed time
class Timer {
public:
    Timer();
    ~Timer();
    /// return the number of wall-clock seconds passed since the creation of the object
    /// (fractional if compiled in C++11 mode, otherwise integer)
    double deltaSeconds() const;
private:
    class Impl;         ///< opaque internal data
    const Impl* impl;   ///< internal object hiding the implementation details
    Timer& operator= (const Timer&);  ///< assignment is disabled
    Timer(const Timer&);              ///< copy constructor is disabled
};

/*------- convenience function -------*/

/// check if a file with this name exists
bool fileExists(const std::string& fileName);

}  // namespace
