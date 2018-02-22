#include "utils_config.h"
#include "utils.h"
#include <fstream>
#include <stdexcept>

namespace utils {

namespace{
/// case-insensitive comparison (7-bit charset)
inline bool comparestr(const std::string& one, const std::string& two)
{
    if(one.size()!=two.size()) return false;
    for(size_t i=0; i<one.size(); i++) {
        char o=one[i];
        if(o>=65&&o<=90) o+=32;
        char t=two[i];
        if(t>=65&&t<=90) t+=32;
        if(o!=t) return false;
    }
    return true;
}
/// these characters signify a comment string
inline bool commentStart(const char c)
{
    return c=='#' || c==';' || c=='/';
}
}

// -------- KeyValueMap -------- //

KeyValueMap::KeyValueMap(const int numParams, const char* const* params) : modified(false)
{
    for(int i=0; i<numParams; i++)
        add(params[i]);
}

KeyValueMap::KeyValueMap(const std::string& params, const std::string& whitespace) :
    modified(false)
{
    std::vector<std::string> arr = splitString(params, whitespace);
    for(unsigned int i=0; i<arr.size(); i++)
        add(arr[i].c_str());
}

void KeyValueMap::add(const char* line)
{
    std::string buffer(line);
    std::string::size_type indx = buffer.find_first_not_of(" \t\n\r");
    if(indx!=std::string::npos)   // get rid of spaces at the beginning of the line
        buffer.erase(0, indx);
    indx = buffer.find('=');
    if(indx!=std::string::npos && indx>0 && !commentStart(buffer[0]))
    {   // key-value pair and not a comment
        std::string key = buffer.substr(0, indx);         // get everything before the '=' character
        buffer.erase(0, indx+1);
        indx = key.find_last_not_of(" \t\n\r");
        if(indx!=std::string::npos && indx+1<key.size())  // and remove trailing spaces (before '=')
            key.erase(indx+1);
        indx = buffer.find_first_not_of(" \t\n\r");       // remove leading spaces after '='
        if(indx!=std::string::npos && indx>0)
            buffer.erase(0, indx);
        indx = buffer.find_last_not_of(" \t\n\r");        // and trailing spaces at the end of the line
        if(indx!=std::string::npos && indx+1<buffer.size())
            buffer.erase(indx+1);
        // do not allow a key to appear twice (there is no way to store and retrieve
        // two separate values, so they are banned outright)
        if(contains(key))
            throw std::runtime_error("KeyValueMap: duplicate value for parameter "+key);
        items.push_back(std::pair<std::string, std::string>(key, buffer));
    } else
        // line without '=' or starting with a comment is not a key-value pair:
        // store with empty key (non-retrievable via get methods), but will be kept when writing
        items.push_back(std::pair<std::string, std::string>("", buffer));
}

bool KeyValueMap::contains(const std::string& key) const
{
    for(unsigned int ik=0; ik<items.size(); ik++)
        if(!items[ik].first.empty() && comparestr(items[ik].first, key))
            return true;
    return false;
}

std::string KeyValueMap::getString(const std::string& key, const std::string& defaultValue) const
{
    for(unsigned int ik=0; ik<items.size(); ik++)
        if(!items[ik].first.empty() && comparestr(items[ik].first, key))
            return items[ik].second;
    return defaultValue;
}

std::string KeyValueMap::getStringAlt(const std::string& key1, 
    const std::string& key2, const std::string& defaultValue) const
{
    std::string result = getString(key1);
    if(result.empty())
        result = getString(key2, defaultValue);
    return result;
}

double KeyValueMap::getDouble(const std::string& key, double defaultValue) const
{
    std::string result = getString(key);
    if(result.empty())
        return defaultValue;
    else
        return toDouble(result);
}

double KeyValueMap::getDoubleAlt(const std::string& key1, 
    const std::string& key2, double defaultValue) const
{
    std::string result = getString(key1);
    if(result.empty())
        return getDouble(key2, defaultValue);
    else
        return toDouble(result);
}

int KeyValueMap::getInt(const std::string& key, int defaultValue) const
{
    std::string result = getString(key);
    if(result.empty())
        return defaultValue;
    else
        return toInt(result);
}

double KeyValueMap::getIntAlt(const std::string& key1, 
    const std::string& key2, int defaultValue) const
{
    std::string result = getString(key1);
    if(result.empty())
        return getInt(key2, defaultValue);
    else
        return toInt(result);
}

bool KeyValueMap::getBool(const std::string& key, bool defaultValue) const
{
    return toBool(getString(key, defaultValue?"True":"False"));
}

std::vector<double> KeyValueMap::getDoubleVector(const std::string& key,
    const std::vector<double>& defaultValues) const
{
    std::string result = getString(key);
    if(result.empty())
        return defaultValues;
    else
        return toDoubleVector(splitString(result, ", "));

}

void KeyValueMap::set(const std::string& key, const std::string& value)
{
    modified = true;  // don't check if the new value is different from the old one
    for(unsigned int ik=0; ik<items.size(); ik++)
        if(!items[ik].first.empty() && comparestr(items[ik].first, key)) {
            items[ik].second = value;
            return;
        }
    items.push_back(std::pair<std::string, std::string>(key, value));  // key not found -- add new
}

void KeyValueMap::set(const std::string& key, const char* value)
{
   set(key, std::string(value));
}

void KeyValueMap::set(const std::string& key, const double value, unsigned int width)
{
    set(key, toString(value, width));
}

void KeyValueMap::set(const std::string& key, const int value)
{
    set(key, toString(value));
}

void KeyValueMap::set(const std::string& key, const unsigned int value)
{
    set(key, toString(value));
}

void KeyValueMap::set(const std::string& key, const bool value)
{
    set(key, toString(value));
}

bool KeyValueMap::unset(const std::string& key)
{
    for(unsigned int ik=0; ik<items.size(); ik++)
        if(!items[ik].first.empty() && comparestr(items[ik].first, key)) {
            items.erase(items.begin()+ik);
            return true;
        }
    return false;
}

std::string KeyValueMap::dump() const
{
    std::string str;
    for(unsigned int i=0; i<items.size(); i++)
        if(items[i].first.empty()) {  // an empty line or a comment
            str += items[i].second + '\n';
        } else {     // a normal key=value entry
            str += items[i].first + '=' + items[i].second + '\n';
        }
    return str;
}

std::string KeyValueMap::dumpSingleLine() const
{
    std::string str;
    for(unsigned int i=0; i<items.size(); i++)
        if(!items[i].first.empty())
            str += items[i].first + '=' + items[i].second + ' ';
    return str;
}

std::vector<std::string> KeyValueMap::keys() const
{
    std::vector<std::string> result;
    for(unsigned int i=0; i<items.size(); i++)
        if(!items[i].first.empty())
            result.push_back(items[i].first);
    return result;
}

// -------- ConfigFile -------- //

ConfigFile::ConfigFile(const std::string& _fileName, bool mustExist) :
    fileName(_fileName)
{
    std::ifstream strm(fileName.c_str());
    if(!strm) {
        if(mustExist)
            throw std::runtime_error("File does not exist: "+_fileName);
        else
            return;  // may be used to create a new ini file
    }
    std::string buffer;
    int secIndex = -1;
    while(std::getline(strm, buffer)) {
        std::string::size_type indx = buffer.find_first_not_of(" \t\n\r");  // skip spaces at the beginning
        std::string::size_type indx1= buffer.find(']');
        if(indx!=std::string::npos && buffer[indx] == '[' && indx1!=std::string::npos && indx1>indx)
        {   // section start - parse section name
            buffer = buffer.substr(indx+1, indx1-indx-1);
            indx = buffer.find_first_not_of(" \t\n\r");
            if(indx!=std::string::npos && indx>0)
                buffer.erase(0, indx);
            indx = buffer.find_last_not_of(" \t\n\r");
            if(indx!=std::string::npos && indx+1<buffer.size())
                buffer.erase(indx+1);
            if(buffer.empty())
                continue;
            // find if this section already existed in the list
            secIndex = -1;
            for(size_t i=0; i<sections.size(); i++)
                if(comparestr(sections[i].first, buffer))
                    secIndex = i;
            if(secIndex<0) { // section not found before, add it
                sections.push_back(std::pair<std::string, KeyValueMap>(buffer, KeyValueMap()));
                secIndex = sections.size()-1;
            }
        } else {  // not a section
            if(sections.empty()) { // no sections has been created yet: add an empty section
                sections.push_back(std::pair<std::string, KeyValueMap>("", KeyValueMap()));
                secIndex = 0;
            }
            sections[secIndex].second.add(buffer.c_str());
        }
    }
    strm.close();
}

ConfigFile::~ConfigFile()
{
    bool modified = false;
    for(unsigned int is=0; is<sections.size(); is++)
        modified |= sections[is].second.isModified();
    if(modified && !fileName.empty() && sections.size()>0) {  // need to save ini file
        std::ofstream strm(fileName.c_str());
        if(!strm) {  // sad but true, we can't do anything else but ignore the problem
            msg(VL_WARNING, "ConfigFile", "Cannot write file "+fileName);
            return;
        }
        for(unsigned int is=0; is<sections.size(); is++) {
            std::string dump = sections[is].second.dump();
            if(!dump.empty()) {
                if(!sections[is].first.empty())   // write the section name in brackets (if present)
                    strm << ("["+sections[is].first+"]\n");
                strm << dump;
                // add an empty line after a section if it did not contain one already
                if(dump.size()>=2 && dump.substr(dump.size()-2) != "\n\n" && is<sections.size()-1)
                    strm << '\n';
            }
        }
    }
}

std::vector<std::string> ConfigFile::listSections() const
{
    std::vector<std::string> list(sections.size());
    for(unsigned int i=0; i<sections.size(); i++)
        list[i] = sections[i].first;
    return list;
}

KeyValueMap& ConfigFile::findSection(const std::string& sec)
{
    for(unsigned int is=0; is<sections.size(); is++)
        if(comparestr(sections[is].first, sec))
            return sections[is].second;
    // not found -- add new section
    sections.push_back(std::pair<std::string, KeyValueMap>(sec, KeyValueMap()));
    return sections.back().second;
}

}; // namespace
