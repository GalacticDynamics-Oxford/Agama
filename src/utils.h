/** \file    utils.h
    \brief   several string formatting and error reporting routines
    \author  EV
    \date    2009-2016
*/
#pragma once
#include <string>
#include <vector>

/** Helper routines for string handling and miscellaneous other tasks.  */
namespace utils {

/* ------- logging routines ------- */
// the following definitions ensure that function name will appear
// with full class qualifier in logmsg calls
#ifdef __GNUC__
#define FUNCNAME __PRETTY_FUNCTION__
#else
#define FUNCNAME __FUNCTION__
#endif

/// level of verbosity for logging messages
enum VerbosityLevel {
    VL_MESSAGE = 0,   ///< information messages that are always printed
    VL_WARNING = 1,   ///< important but non-critical warnings indicating possible problems
    VL_DEBUG   = 2,   ///< ordinary debugging messages
    VL_VERBOSE = 3,   ///< copious amount of debugging messages
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

/*------------- string functions ------------*/

/// convert a string to a number
int toInt(const char* val);
float toFloat(const char* val);
double toDouble(const char* val);
// handy overloads
inline int toInt(const std::string& val)
{ return toInt(val.c_str()); }
inline float toFloat(const std::string& val)
{ return toFloat(val.c_str()); }
inline double toDouble(const std::string& val)
{ return toDouble(val.c_str()); }

/// convert a string to a bool
bool toBool(const char* val);
inline bool toBool(const std::string& val)
{ return toBool(val.c_str()); }

/// convert a bool to a string
inline static std::string toString(bool val)
{ return val?"true":"false"; }

/// convert a number to a string with a given precision (number of significant digits)
std::string toString(double val, unsigned int width=6);
std::string toString(float val, unsigned int width=6);
std::string toString(int val);
std::string toString(unsigned int val);
/// convert a pointer to a string
std::string toString(const void* val);
/// convert any typed pointer to a string
template<typename T> inline std::string toString(const T* val) {
    return toString(static_cast<const void*>(val));
}

/// Pretty-print: convert floating-point or integer numbers to a string of exactly fixed length.
/// Choose fixed-point or exponential format depending on which one is more accurate;
/// if the number does not fit into the given width, return a string with # symbols.
std::string pp(double num, unsigned int width);

/** routine that splits one line from a text file into several items.
    \param[in]  src -- string to be split;
    \param[in]  delim -- array of characters used as delimiters;
    \return  an array of strings
*/
std::vector<std::string> splitString(const std::string& src, const std::string& delim);

/// routine that checks if a string ends with another string
bool endsWithStr(const std::string& str, const std::string& end);

/// routine that compares two strings in a case-insensitive way
bool stringsEqual(const std::string& str1, const std::string& str2);

/// overloaded routine that compares two strings in a case-insensitive way
bool stringsEqual(const std::string& str1, const char* str2);

}  // namespace
