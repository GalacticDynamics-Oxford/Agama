#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// stack trace presumably only works with GCC
#ifdef __GNUC__
#include <cxxabi.h>
#include <execinfo.h>
#endif

namespace utils {

bool fileExists(const std::string& fileName)
{
    std::ifstream infile(fileName.c_str());
    return infile.good();
}

/* -------- error reporting routines ------- */

namespace{  // internal

/// remove function signature from GCC __PRETTY_FUNCTION__
inline std::string undecorateFunction(std::string origin)
{
#ifdef __GNUC__
    // parse the full function signature returned by __PRETTY_FUNCTION__
    std::string::size_type ind=origin.find('(');
    if(ind!=std::string::npos)
        origin.erase(ind);
    ind=origin.rfind(' ');
    if(ind!=std::string::npos)
        origin.erase(0, ind+1);
#endif
    return origin;
}

/// check if a character is a part of a valid C++ identifier name
inline bool isAlphanumeric(const char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

/// attempt to demangle any possible C++ identifier names that are encountered in the input string
std::string demangleName(char* const input)
{
    std::string result;
    char *ptr = input, *name = NULL;
    while(1) {                       // scan the entire input line until '\0' is found.
        if(isAlphanumeric(*ptr)) {   // current character is a part of an identifier name:
            if(!name)                // if we are not already in the regime of recording a name,
                name = ptr;          // turn this regime on and remember the start of the name.
        } else {                     // otherwise the character is not part of a name:
            if(name) {               // if we have been recording a name, finalize this name.
                char oldsym = *ptr;  // remember the character at the current position
                *ptr = '\0';         // temporarily construct a null-terminated string with the name
                int status;          // try to demangle an identifier name
                char* demangledName = abi::__cxa_demangle(name, NULL, NULL, &status);
                if(status == 0) {    // success: append the demangled name to the results
                    result.append(demangledName);
                } else {             // demangling failed: append the name itself
                    result.append(name);
                }
                free(demangledName); // free the temporary string returned by __cxa_demangle()
                *ptr = oldsym;       // put the current character back into place
            }
            name = NULL;             // turn off the regime of recording a name
            if(*ptr != '\0')         // append the current character (not part of any name) to results
                result.append(1, *ptr);
        }
        if(*ptr == '\0')             // reached the end of the input string
            break;
        else
            ++ptr;                   // pass on to the next character of the input string
    }
    return result;
}

/// a prefix character prepended to the output message
inline char verbosityText(VerbosityLevel level)
{
    switch(level) {
        case VL_MESSAGE: return '.';
        case VL_WARNING: return '!';
        case VL_DEBUG:   return '-';
        case VL_VERBOSE: return '=';
        default: return '?';
    }
}

/// file to store the messages sent to msg() routine;
/// if not open, they are printed to stderr
static std::ofstream logfile;
    
/// read the environment variables controlling the verbosity level and log file redirection
VerbosityLevel initVerbosityLevel()
{
    const char* env = std::getenv("LOGFILE");
    if(env) {
        logfile.open(env);
    }
    env = std::getenv("LOGLEVEL");
    if(env && env[0] >= '0' && env[0] <= '3')
        return static_cast<VerbosityLevel>(env[0]-'0');
    return VL_MESSAGE;  // default
}

/// default routine that dumps text messages to stderr
void defaultmsg(VerbosityLevel level, const char* origin, const std::string &message)
{
    if(level > verbosityLevel)
        return;
    std::string msg = verbosityText(level) + 
        (origin==NULL || *origin=='\0' ? "" : "{"+undecorateFunction(origin)+"} ") + message + '\n';
    if(logfile.is_open())
        logfile << msg << std::flush;
    else
        std::cerr << msg << std::flush;
}

}  // namespace

/// global pointer to the routine that displays or logs information messages
MsgType* msg = &defaultmsg;

/// global variable controlling the verbosity of printout
VerbosityLevel verbosityLevel = initVerbosityLevel();

std::string stacktrace()
{
#ifdef __GNUC__
    const int MAXFRAMES = 256;
    void* tmp[MAXFRAMES];     // temporary storage for stack frames
    int numframes = backtrace(tmp, MAXFRAMES);
    // convert this information into strings with function signatures and offsets
    char** lines = backtrace_symbols(tmp, numframes);  // this array should be freed later
    // represent these strings in a more user-friendly way (demangle identifier names)
    std::string result = "Stack trace ("+toString(numframes)+
        " functions, starting from the most recent one):\n";
    for(int i=1; i<numframes; i++)
        result += demangleName(lines[i]) + '\n';
    free(lines);
    return result;
#else
    return "Stack trace not available\n";
#endif
}

/* ----------- string/number conversion and parsing routines ----------------- */

int toInt(const char* val) {
    return strtol(val, NULL, 10);
}

float toFloat(const char* val) {
    return strtof(val, NULL);
}

double toDouble(const char* val) {
    return strtod(val, NULL);
}

std::string toString(double val, unsigned int width) {
    char buf[100];
    snprintf(buf, 100, "%-.*g", width, val);
    return std::string(buf);
}

std::string toString(float val, unsigned int width) {
    char buf[100];
    snprintf(buf, 100, "%-.*g", width, val);
    return std::string(buf);
}

std::string toString(int val) {
    char buf[100];
    snprintf(buf, 100, "%i", val);
    return std::string(buf);
}

std::string toString(unsigned int val) {
    char buf[100];
    snprintf(buf, 100, "%u", val);
    return std::string(buf);
}

std::string toString(long val) {
    char buf[100];
    snprintf(buf, 100, "%li", val);
    return std::string(buf);
}

std::string toString(unsigned long val) {
    char buf[100];
    snprintf(buf, 100, "%lu", val);
    return std::string(buf);
}

std::string toString(const void* val) {
    char buf[100];
    snprintf(buf, 100, "%p", val);
    return std::string(buf);
}

bool toBool(const char* val) {
    return
        strncmp(val, "yes", 3)==0 ||
        strncmp(val, "Yes", 3)==0 ||
        strncmp(val, "true", 4)==0 ||
        strncmp(val, "True", 4)==0 ||
        strncmp(val, "t", 1)==0 ||
        strncmp(val, "1", 1)==0;
}

//  Pretty-print - convert float (and integer) numbers to string of fixed width.
//  Employ sophisticated techniques to fit the number into a string of exactly the given length.
std::string pp(double num, unsigned int uwidth)
{
    const int MAXWIDTH = 31;  // rather arbitrary restriction, but doubles never are that long
    std::string result;
    int width = std::min<int>(uwidth, MAXWIDTH);
    if(width<1)
        return result;
    unsigned int sign = num<0;
    if(num==0) {  // no difference between +0 and -0
        result = "0";
        if(width>1) result+='.';
        result.resize(width, '0');
        return result;
    }
    if(num!=num || num==INFINITY || num==-INFINITY) {
        result = num==INFINITY ? "+INF" : num==-INFINITY ? "-INF" : "NAN";
        result.resize(width, '#');
        return result;
    }
    // separate out sign, and reduce the available width
    if(sign) {
        result = "-";
        width--;
    }
    if(width==0) {
        return result;
    }
    num = fabs(num);

    // decimal exponent
    int expon = (int)floor(log10(num));
    char buf[MAXWIDTH+1];

    // now we have three possibilities:
    // 1) exponential notation:  2.34e5 or 6e-7 - at least one digit before the 'e' symbol
    // 2) fixed-point notation:  234321.2 or 0.0000006321
    // 3) "#" if none of the above fits into the available space

    // try to print the number x>=1 in fixed-point format
    if(expon >= 0 && expon <= width-1) {
        int len = snprintf(buf, MAXWIDTH, "%-#.*f", std::max<int>(width-2-expon, 0), num);
        if(len == width+1 && buf[width] == '.') {
            // exceeds the required width, but the final decimal point may be removed
            buf[width] = '\0';
            len--;
        }
        if(len == width) {
            result += buf;
            return result;
        }
        // otherwise we may have encountered the situation of rounding up
        // a number 9.x to 10. and exceeding the width
    }

    // expected length of the exponent part of the string if we choose exponential notation
    // (including the 'e' symbol, and removing the sign of exponent if it is positive)
    int  len_exp = expon<=-100 ? 5 : expon<=-10 ? 4 : expon<0 ? 3 : expon<10 ? 2 : expon<100 ? 3 : 4;

    // expected # of significant digits in mantissa in the exp format (including the leading digit)
    int  dig_exp = std::max<int>(width-len_exp-1, 1);  // (one position is occupied by decimal point)

    // expected number of significant digits in fixed-point format if |x|<1
    // (first two symbols are "0.", and then possibly a few more zeros before the number begins)
    int  dig_lt1 = width-1+expon;  // e.g. if expon=-2 and width=6, we may have 0.0234 - 3 digit accuracy

    // try to print the number x<1 in fixed-point format if the expected precision is no less than
    // in the exponential format, or in the special case of a number 0.5<=x<1 rounded up to 1.
    if(expon < 0 && (dig_lt1 >= dig_exp || (num>=0.5 && width<=2))) {
        int len = snprintf(buf, MAXWIDTH, "%-#.*f", std::max<int>(width-2, 0), num);
        if(len > width) {
            buf[width] = '\0';
            len--;
        }
        result += buf;
        return result;
    }
    
    // try to print the number in exponential format
    if(width >= len_exp+1) {
        // strip out exponent, so that the number is within [1:10)
        num = fmax(num / pow(10., expon), 1.);  // it might be <1 due to roundoff error
        int len = snprintf(buf, MAXWIDTH, "%-#.*f", std::max<int>(width-2-len_exp, 0), num);
        if(len >= 2 && buf[0] == '1' && buf[1] == '0') {
            // a number 9.x is rounded up to 10, so we should replace it with 1. and increase the exponent
            expon++;
            buf[1] = '.';
            if(len>2)
                buf[2] = '0';
        }
        char buf_exp[8] = {'e'};
        len_exp = snprintf(buf_exp+1, 6, "%-i", expon)+1;
        if(len_exp < width) {
            if(len > width-len_exp)
                buf[width-len_exp] = '\0';
            result += buf;
            result += buf_exp;
            return result;
        }
    }
    
    // can't use any of these - display a jail symbol
    result.resize(sign+width, '#');
    return result;
}

std::vector<std::string> splitString(const std::string& src, const std::string& delim)
{
    std::vector<std::string> result;
    std::string str(src);
    std::string::size_type indx=str.find_first_not_of(delim);
    if(indx==std::string::npos) {
        result.push_back("");   // ensure that result contains at least one element
        return result;
    }
    if(indx>0)  // remove unnecessary delimiters at the beginning
        str=str.erase(0, indx);
    while(!str.empty()) {
        indx=str.find_first_of(delim);
        if(indx==std::string::npos)
            indx=str.size();
        result.push_back(str.substr(0, indx));
        str=str.erase(0, indx);
        indx=str.find_first_not_of(delim);
        if(indx==std::string::npos)
            break;
        str=str.erase(0, indx);
    }
    return result;
}

bool endsWithStr(const std::string& str, const std::string& end)
{
    return end.size()<=str.size() && str.find(end, str.size()-end.size())!=str.npos;
}

bool stringsEqual(const std::string& str1, const std::string& str2)
{
    std::string::size_type len=str1.size();
    if(len!=str2.size())
        return false;
    for(std::string::size_type i=0; i<len; i++)
        if(tolower(str1[i]) != tolower(str2[i]))
            return false;
    return true;
}

bool stringsEqual(const std::string& str1, const char* str2)
{
    if(str2==NULL)
        return false;
    for(std::string::size_type i=0; i<str1.size(); i++)
        if(str2[i]==0 || tolower(str1[i]) != tolower(str2[i]))
            return false;
    return str2[str1.size()]==0;  // ensure that the 2nd string length is the same as the 1st
}

}  // namespace
