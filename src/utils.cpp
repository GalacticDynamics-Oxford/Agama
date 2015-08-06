#include "utils.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace utils {

#if 0
/* -------- error reporting routines ------- */

/// remove common prefix if it is the namespace "smile::", and remove function signature from GCC __PRETTY_FUNCTION__
std::string undecorateFunction(std::string origin)
{
#ifdef __GNUC__
    // parse the full function signature returned by __PRETTY_FUNCTION__
    std::string::size_type ind=origin.find('(');
    if(ind!=std::string::npos) origin.erase(ind);
    ind=origin.rfind(' ');
    if(ind!=std::string::npos) origin.erase(0, ind+1);
#endif
    if(origin.substr(0, 7)=="smile::") origin.erase(0, 7);
    std::string::size_type indc=origin.find(':');
    if(indc!=std::string::npos && indc+1==origin.size()/2 && origin[indc+1]==':') {
        if(origin.substr(0, indc)==origin.substr(indc+2)) origin.replace(indc+2, std::string::npos, "Ctor");
        if(origin[indc+2]=='~' && origin.substr(0, indc)==origin.substr(indc+3)) origin.replace(indc+2, std::string::npos, "Dtor");
    }
    return origin;
}

/// default routine that dumps text messages to stderr
void my_stderr_show_message(const std::string &origin, const std::string &message)
{
    std::cerr << ((origin.empty() ? "" : "{"+undecorateFunction(origin)+"} ") + message+"\n"); 
}

/// global variable to the routine that displays errors (if it is NULL then nothing is done)
show_message_type* my_error_ptr   = &my_stderr_show_message;

/// global variable to the routine that shows information messages 
/// (emitted by various time-consuming routines to display progress), if NULL then no messaging is done
show_message_type* my_message_ptr = &my_stderr_show_message;

#endif

#if 0
// old version, slow
    
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

int convertToInt(const char* val) {
    return StringVariant(val).toNumber<int>();
}
float convertToFloat(const char* val) {
    return StringVariant(val).toNumber<float>();
}
double convertToDouble(const char* val) {
    return StringVariant(val).toNumber<double>();
}
template<typename ValueType> std::string convertToString(ValueType val) { 
    return StringVariant(val).toString();
}
template<typename ValueType> std::string convertToString(ValueType val, unsigned int width) { 
    return StringVariant(val,width).toString(); }

#else
// old-fashioned way, faster

int convertToInt(const char* val) {
    return strtol(val, NULL, 10);
}
float convertToFloat(const char* val) {
    return strtod(val, NULL);
}
double convertToDouble(const char* val) {
    return strtod(val, NULL);
}
template<typename ValueType> std::string convertToString(ValueType val, unsigned int width=6) {
    char buf[100];
    int len=snprintf(buf, 100, "%*g", width, val);
    int offset=0;
    while(offset<len && buf[offset]==' ') offset++;
    return std::string(buf+offset);
}
template<> std::string convertToString(int val, unsigned int width) {
    char buf[100];
    int len=snprintf(buf, 100, "%*i", width, val);
    int offset=0;
    while(offset<len && buf[offset]==' ') offset++;
    return std::string(buf+offset);
}
template<> std::string convertToString(unsigned int val, unsigned int width) {
    char buf[100];
    int len=snprintf(buf, 100, "%*u", width, val);
    int offset=0;
    while(offset<len && buf[offset]==' ') offset++;
    return std::string(buf+offset);
}
#endif

template std::string convertToString(int, unsigned int);
template std::string convertToString(float, unsigned int);
template std::string convertToString(double, unsigned int);

bool convertToBool(const char* val) {
    return 
        strncmp(val, "yes", 3)==0 ||
        strncmp(val, "Yes", 3)==0 ||
        strncmp(val, "true", 4)==0 ||
        strncmp(val, "True", 4)==0 ||
        strncmp(val, "t", 1)==0 ||
        strncmp(val, "1", 1)==0;
}

/* ----------- string/number conversion and parsing routines ----------------- */
//  Pretty-print - convert float (and integer) numbers to string of fixed width.
//  Uses some sophisticated techniques to fit the number into a string of exactly the given length.
std::string pp(double num, unsigned int width)
{
    std::string result, tmp;
    if(num==0) { 
        for(int i=0; i<static_cast<int>(width)-1; i++) result+=' ';
        result+='0';
        return result;
    }
    unsigned int sign=num<0;
    double mag=log10(fabs(num));
    std::ostringstream stream;
    if(num!=num || num/2==num || num+0!=num)
    {
        if(width>=4) stream << std::setw(width) << num;
        else stream << "#ERR";
    }
    else if(width<=2+sign)  // display int if possible
    {
        if(mag<0) stream << (sign?"-":"+") << 0;
        else if(mag>=2-sign) stream << (sign?"-":"+") << "!";
        else stream << (int)floor(num+0.5);
    }
    else if(mag>=0 && mag+sign<width && mag<6)  // try fixed-point for |x|>=1
    {
        stream << std::setw(width) << std::setprecision(width-1-sign) << num;
        if(stream.str().find('e')!=std::string::npos) { 
            //std::string x=stream.str();
            //size_t e=x.find('e');
            stream.str(""); 
            stream << (int)floor(num+0.5); 
        }
    }
    else if(mag<0 && -mag+sign<width-2 && mag>=-4) // try fixed-point for |x|<1
    {
        stream << std::setw(width) << std::setprecision(width-1-sign+(int)floor(mag)) << num;
    }
    else
    {
        std::ostringstream strexp;
        int expon=static_cast<int>(floor(mag));
        strexp << std::setiosflags(std::ios_base::showpos) << expon;
        std::string expstr=strexp.str();
        size_t w=(width-expstr.size()-1);
        double mant=num*pow(10.0, -expon);
        if(w<sign)  // no luck with exp-format -- try fixed 
        {
            stream << (sign?"-":"+") << (mag<0 ? "0" : "!");
        }
        else 
        {
            if(w==sign) 
                stream << (sign?"-":""); // skip mantissa
            else if(w<=2+sign)
            { 
                int mantint=(int)floor(fabs(mant)+0.5);
                if(mantint>=10) mantint=9;  // hack
                stream << (sign?"-":"") << mantint;
            }
            else
                stream << std::setprecision(w-1-sign) << mant;
            stream << "e" << expstr;
        }
    }
    result=stream.str();
    // padding if necessary (add spaces after string)
    while(result.length()<static_cast<size_t>(width))
        result+=" ";
    if(result.length()>static_cast<size_t>(width))  // cut tail if necessary (no warning given!)
        result=result.substr(0,width);
    return result;
}

void splitString(const std::string& src, const std::string& delim, std::vector<std::string>& result)
{
    result.clear();
    std::string str(src);
    std::string::size_type indx=str.find_first_not_of(delim);
    if(indx==std::string::npos) {
        result.push_back("");   // ensure that result contains at least one element
        return;
    }
    if(indx>0)  // remove unnecessary delimiters at the beginning
        str=str.erase(0, indx);
    while(!str.empty())
    {
        indx=str.find_first_of(delim);
        if(indx==std::string::npos)
            indx=str.size();
        result.push_back(str.substr(0, indx));
        str=str.erase(0, indx);
        indx=str.find_first_not_of(delim);
        if(indx==std::string::npos)
            return;
        str=str.erase(0, indx);
    }
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
