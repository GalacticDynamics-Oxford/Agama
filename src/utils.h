/** \file    utils.h
    \brief   several string formatting and error reporting routines
    \author  EV
    \date    2009-2014

*/
#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace utils {
#if 0
/* ------- message and error reporting routines ------- */
// the following definitions ensure that function name will appear with full class qualifier in my_message calls
#ifdef __GNUC__
#define FUNCNAME __PRETTY_FUNCTION__
#else
#define FUNCNAME __FUNCTION__
#endif

/// a function that shows a given message string.
/// implementation is program-specific, may dump text to stderr or do something else.
/// \param[in] origin is the name of calling function
/// \param[in] message is the text to be displayed
typedef void show_message_type(const std::string &origin, const std::string &message);

/// global variable to the routine that displays errors (if it is NULL then nothing is done)
extern show_message_type* my_error_ptr;

/// global variable to the routine that shows information messages.
/// called by various time-consuming routines to display progress, if NULL then no messaging is done
extern show_message_type* my_message_ptr;

/// the interface routine for error reporting (redirects the call to my_error_ptr if it is defined)
inline static void my_error(const std::string &origin, const std::string &message)
{ if(my_error_ptr!=NULL) my_error_ptr(origin, message); }

/// the interface routine for message reporting (redirects the call to my_message_ptr if it is defined)
inline static void my_message(const std::string &origin, const std::string &message)
{ if(my_message_ptr!=NULL) my_message_ptr(origin, message); }

/// default message printing routine
void my_stderr_show_message(const std::string &origin, const std::string &message);
#endif

/*------------- string functions ------------*/
/** The StringVariant class is a simple string-based variant implementation that allows
    the user to easily convert between simple numeric/string types.
*/
class StringVariant
{
private:
    std::string data;
public:
    StringVariant() : data() {}
    StringVariant(const std::string &src) : data(src) {};
    template<typename ValueType> StringVariant(ValueType val) {
        std::ostringstream stream;
        stream << val;
        data.assign(stream.str());
    };
    template<typename ValueType> StringVariant(ValueType val, unsigned int width) {
        std::ostringstream stream;
        stream << std::setprecision(width) << val;
        data.assign(stream.str());
    };
    template<typename ValueType> StringVariant& operator=(const ValueType val) {
        std::ostringstream stream;
        stream << val;
        data.assign(stream.str());
        return *this;
    };
    template<typename NumberType> NumberType toNumber() const {
        NumberType result = 0;
        std::istringstream stream(data);
        if(stream >> result)
            return result;
        else if(data == "yes" || data == "true")
            return 1;
        return 0;
    };
    bool toBool() const { return(data == "yes" || data == "Yes" || data == "true" || data == "True" || data == "t" || data == "1"); }
    double toDouble() const { return toNumber<double>(); }
    float toFloat() const { return toNumber<float>(); }
    int toInt() const { return toNumber<int>(); }
    std::string toString() const { return data; }
};

/// a shorthand function to convert a string to a number
template<typename NumberType, typename ValueType> NumberType convertTo(ValueType val) 
{ return StringVariant(val).toNumber<NumberType>(); }

/// a shorthand function to convert a string to a bool
template<typename ValueType> bool convertToBool(ValueType val) 
  { return StringVariant(val).toBool(); }

/// a shorthand function to convert a number to a string
template<typename ValueType> std::string convertToString(ValueType val) 
  { return StringVariant(val).toString(); }

/// a shorthand to convert a bool to a string
inline static const char* convertToString(bool val)
  { return val?"true":"false"; }

/// a shorthand function to convert a number to a string with a given precision
template<typename ValueType> std::string convertToString(ValueType val, unsigned int width) 
  { return StringVariant(val,width).toString(); }

/// Pretty-print: convert floating-point or integer numbers to a string of fixed length.
std::string pp(double num, unsigned int width);

/** routine that splits one line from a text file into several items.
    \param[in]  src -- string to be split
    \param[in]  delim -- array of characters used as delimiters
    \param[out] result -- pointer to an existing array of strings in which the elements will be stored
*/
void splitString(const std::string& src, const std::string& delim, std::vector<std::string> *result);

/// routine that checks if a string ends with another string
bool ends_with_str(const std::string& str, const std::string& end);

/// routine that compares two strings in a case-insensitive way
bool strings_equal(const std::string& str1, const std::string& str2);

/// overloaded routine that compares two strings in a case-insensitive way
bool strings_equal(const std::string& str1, const char* str2);

}  // namespace
