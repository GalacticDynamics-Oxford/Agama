/** \file    utils_config.h
    \brief   Handling of configuration data in INI files
    \author  Eugene Vasiliev
    \date    2014-2015

    This file defines the class for reading and writing parameters from/to INI text files.
    The file format is identical to MS .ini file: 

    [Section]
    key=value
    ;key=old value -- commented out,
    //lines starting with ; # / are ignored but preserved in the file upon saving

*/

#pragma once
#include <string>
#include <vector>
#include <utility>

namespace utils {

/** Class that stores parameters in the form of key/value pairs, with case-insensitive key lookup
    and preservation of the order of elements  */
class KeyValueMap {
public:
    /// initialize an empty map
    KeyValueMap() : modified(false) {};

    /// initialize map from command-line-like arguments
    KeyValueMap(const int numParams, const char* const* params);

    /// initialize map from a single string in which all parameters are listed together,
    /// separated by any of white characters listed in the second argument
    explicit KeyValueMap(const std::string& params, const std::string& whitespace=", \t");

    /// check if a key exists in the list
    bool contains(const std::string& key) const;

    /// return a string value from the map, or defaultValue if the key is not found in the list
    std::string getString(const std::string& key, const std::string& defaultValue="") const;

    /// return value from either of the two variants of key
    std::string getStringAlt(const std::string& key1, const std::string& key2,
        const std::string& defaultValue="") const;

    /// return a floating-point value from the map
    double getDouble(const std::string& key, double defaultValue=0) const;

    /// return a float from either of the two variants of key
    double getDoubleAlt(const std::string& key1, const std::string& key2,
        double defaultValue=0) const;

    /// return an integer value from the map
    int getInt(const std::string& key, int defaultValue=0) const;

    /// return an integer from either of the two variants of key
    int getIntAlt(const std::string& key1, const std::string& key2,
        int defaultValue=0) const;

    /// return a boolean value from the map
    bool getBool(const std::string& key, bool defaultValue=false) const;

    /// return a boolean value from either of the two variants of key
    bool getBoolAlt(const std::string& key1, const std::string& key2,
        bool defaultValue=false) const;

    /// return an array of floating-point values parsed from a comma-separated string
    std::vector<double> getDoubleVector(const std::string& key,
        const std::vector<double>& defaultValues) const;

    /// set a string value
    void set(const std::string& key, const std::string& value);

    /// set a string value
    void set(const std::string& key, const char* value);

    /// set a floating-point value
    void set(const std::string& key, const double value, unsigned int width=6);

    /// set an integer value
    void set(const std::string& key, const int value);

    /// set an integer value
    void set(const std::string& key, const unsigned int value);

    /// set a boolean value
    void set(const std::string& key, const bool value);

    /// attempt to delete a key from the list; return true if it existed
    bool unset(const std::string& key);

    /// parse a key=value pair and append it to the map (does not change `modified` flag)
    void add(const char* keyValue);

    /// dump the entire map into an array of strings, retaining all entries
    /// (including comments and empty lines)
    std::vector<std::string> dumpLines() const;

    /// dump the entire map into a single line, omitting comments
    /// and joining entries with a space character
    std::string dumpSingleLine() const;

    /// return the list of all keys that contain any values
    std::vector<std::string> keys() const;

    /// check if any items were modified by a call to set method
    bool isModified() const { return modified; }

private:
    std::vector< std::pair< std::string, std::string > > items;
    bool modified;  ///< keeps track of manual changes made by set method
};

/** Utility class for handling an INI file.
    Provides high-level interface for reading and writing values of various types, which are stored
    in a text format in one or more sections.
*/
class ConfigFile {
public:
    /// open an INI file with the given name and read its contents
    /// \throw std::runtime_error if the file does not exist and mustExist==true
    ConfigFile(const std::string& fileName, bool mustExist=true);

    /// destroy the class and write unsaved data back to INI file, if it was modified
    /// (that is, if a 'set()' method was called for any of the sections)
    ~ConfigFile();

    /// return a list of all sections in the INI file
    std::vector<std::string> listSections() const;

    /// find the section by its name.
    /// \param[in]  secName  is the section name; if it did not exist, first create an empty section.
    /// \return a non-const reference to a section: it may be used to modify the values in the ini file
    KeyValueMap& findSection(const std::string& secName);
    const KeyValueMap& findSection(const std::string& secName) const;

private:
    /// name of the INI file
    std::string fileName;
    /// list of all sections with their names
    std::vector< std::pair<std::string, KeyValueMap> > sections;
};

} // namespace
