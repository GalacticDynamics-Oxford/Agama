/** \file    test_utils.cpp
    \date    2016-2017
    \author  Eugene Vasiliev

    Test the number-to-string conversion routine and the ini-file manipulation routines
*/
#include "utils.h"
#include "utils_config.h"
#include "math_core.h"
#include "math_random.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

bool testNumberConversion()
{
    // this should pass, ignoring whitespace and trailing non-numeric characters
    try{ if(math::fcmp(utils::toDouble(" -9.e99x y"), -9e99, 1e-15) != 0) return false; }
    catch(std::exception&) { return false; }
    // this should fail with an exception
    try{ utils::toFloat("x789.5"); return false; }
    catch(std::exception&) {}
    // this should pass and return an integer part of the number
    try{ if(utils::toInt("\t -9.99e1 # -99.9") != -99) return false; }
    catch(std::exception&) { return false; }
    // infinity or NAN should be parsed normally
    try{ if(utils::toDouble(" -Infinit###") != -INFINITY) return false; }
    catch(std::exception&) { return false; }
    try{ if(isFinite(utils::toDouble("NaNcrazy"))) return false; }
    catch(std::exception&) { return false; }
    // ..but not if the return value is integer
    try{ utils::toInt("INFINITY"); return false; }
    catch(std::exception&) {}
    try{ utils::toInt("NAN"); return false; }
    catch(std::exception&) {}

    // continue in a normal way, testing the fixed-width number-to-string conversion routine
    bool ok=true;
    const int NUM = 50;
    double values[NUM] = {0, NAN, INFINITY, -INFINITY, 1., 
       9.5e-5, 0.95000000000001, 9.4999999999999, 9.95, 999.5, 9999.5, 9.5e8, 9.96e9};
    for(int i=13; i<NUM; i++) {
        double val = math::random();
        int type   = (int)(math::random()*8);
        if(type & 1)
            val = pow(10, 10*val);
        else
            val = pow(10, 150*val);
        if(type & 2)
            val = 1/val;
        if(type & 4)
            val = -val;
        values[i] = val;
    }
    for(int i=0; i<NUM; i++) {
        std::cout << std::setw(24) << std::setprecision(16) << values[i];
        // represent the same number with strings of different width
        for(unsigned int w=1; w<12; w++) {
            std::string result = utils::pp(values[i], w);
            bool fail  = result.size() != w;  // should be exactly the desired length
            double val = 0;
            try {
                // try to parse the string
                val = utils::toDouble(result);
                // if it succeeded, compare the value very roughly
                fail |= ((val>0) ^ (values[i]>0)) ||  // not the same sign
                val / values[i] < 0.5 || val / values[i] > 2;  // differs by more than a factor of 2
            }
            catch(std::exception&) {  // could not parse the string - may be for a legitimate reason
                if(values[i] != values[i]) {  // NAN cannot be represented by less than 3 characters
                    fail |= w>3;
                } else if(!isFinite(values[i])) {  // signed infinity needs at least 4 characters
                    fail |= w>4;
                } else if(values[i] > 0) {
                    // a finite positive number cannot be parsed if it was rendered as "#"
                    fail |= result[0] != '#';
                } else if(values[i] < 0) {
                    // a finite negative number cannot be parsed if it was rendered as "-#"
                    fail |= result[0] != '-';
                    if(w>1) fail |= result[1] != '#';
                } else  // the only remaining variant is if v=0, and it should have been parsed
                    fail = true;
            }
            std::cout << ' ';
            if(fail) std::cout << "\033[1;31m";
            std::cout << result;
            if(fail) std::cout << "\033[0m";
            ok &= !fail;
        }
        std::cout << '\n';
    }
    return ok;
}

bool testSplitString()
{
    std::vector<std::string> fields;
    fields = utils::splitString("", "|");
    if(!fields.empty())
        return false;
    fields = utils::splitString(" \t \t ", " \t");
    if(!fields.empty())
        return false;
    fields = utils::splitString(" item1 item2 \t item3\t\t", "\t ");
    if(! (fields.size()==3 && fields[0]=="item1" && fields[1]=="item2" && fields[2]=="item3"))
        return false;
    return true;
}

std::string readFile(const char* filename)
{   // read the file into a single string
    std::ifstream in(filename);
    std::string buffer, result;
    while(std::getline(in, buffer))
        result += buffer+'\n';
    return result;
}

bool testIniFile()
{
    // an assorted selection of traps to be parsed as an INI file
    const char* iniFile =
    // lines starting with '#' are comments, they are preserved upon saving
    "## comment ##\n"
    // extra spaces at the beginning of line are erased; anything inside a comment is ignored
    "  ## another comment [pretends= to be a section!] ## \n"
    // empty lines don't mean anything but are preserved upon saving
    "\n"
    // before a [section] block is encountered, anything is attributed to the empty section (preserved)
    "param_without_section=value\n"
    // now the first section begins
    "[Section1]\n"
    // ordinary parameter=value
    "par1sec1=value1\n"
    // section names have leading and trailing spaces which are ignored
    " [ ## section2 ## ]\n"
    // leading and trailing spaces before and after key and value are removed;
    // here the value starts with '=' and contains various special characters that are considered normal
    "   par1sec2  == value 2 // [comment?] // \n"
    // a comment line may also start with ';', special characters '=', '[]' are ignored
    "   ; another innocent comment = nothing[?] \n"
    // a key may contain comment signs if it doesn't start with a comment
    "pa#ram2 sec2 = 1.5  # this is part of the value!\n"
    // lines starting with '/' are also comments
    "// ok enough\n"
    // the same section again; extra rubbish after its name is ignored
    "[ section1 ] # again!\n"
    // comment sign inside a value is not a comment anymore
    " par2sec1  =   #?@$%^&*!!!\n"
    // a line without a '=' sign is not retrievable via contains or get*** methods
    "nonexistent\n";

    const char* tmpfilename = "tmp.ini";
    bool ok=true;

    {   // write out an ini file with random data
        std::ofstream out(tmpfilename);
        out << iniFile;
    }
    {   // parse the file using the ConfigFile class and change a value in one of its sections
        utils::ConfigFile cfg(tmpfilename);
        utils::KeyValueMap& sec1 = cfg.findSection("section1");
        // simply check that the value is retrieved from a case-insensitive key
        ok &= sec1.getString("PAR1SEC1") == "value1";
        // set a new value to the same key (this forces the ini file to be saved)
        sec1.set("par1SEC1", 2.71);
        // parse a string with non-numeric characters as a number (the remaining part of line is ignored)
        ok &= cfg.findSection("## section2 ##").getDouble("pa#ram2 sec2") == 1.5;
        // find a parameter in a second part of the first section, but not a line without a '=' character
        ok &= sec1.contains("par2sec1") && !sec1.contains("nonexistent");
        // extra leading/trailing spaces should have been ignored for both the key and its value
        ok &= cfg.findSection("SECTION1").getString("par2sec1") == "#?@$%^&*!!!";
        // add a parameter
        sec1.set("new param", "some value");
        // set an entry without a key which will be ignored upon reading
        sec1.set("", "injected line");
        // the ini file should be saved when the instance of ConfigFile is destroyed
    }
    // read the updated file into a single string
    std::string iniFile1 = readFile(tmpfilename);
    {   // yet another read-and-modify
        utils::ConfigFile cfg(tmpfilename);
        // modify the value in the unnamed (0th) section
        cfg.findSection("").set("param_without_section", "  new value  ");
        // retrieve the previously assigned value as an integer (this ignores the fractional part)
        ok &= cfg.findSection("Section1").getInt("par1sec1") == 2;
        // retrieve the previously added parameter
        ok &= cfg.findSection("Section1").getString("new param") == "some value";
        // no key - not retrievable
        ok &= !cfg.findSection("section1").contains("injected line");
    }
    {   // still another read-and-modify
        utils::ConfigFile cfg(tmpfilename);
        // no extra leading/trailing spaces in the value
        ok &= cfg.findSection("").getString("param_without_section") == "new value";
        // assign it back to the original value
        cfg.findSection("").set("param_without_section", "value");
    }
    std::string iniFile2 = readFile(tmpfilename);
    ok &= iniFile1 == iniFile2;
    ok &=
        iniFile1.find("innocent comment") != std::string::npos &&
        iniFile1.find("injected") != std::string::npos &&
        iniFile1.find("par2sec1=#?@$%^&*!!!") != std::string::npos &&
        iniFile1.substr(0, 13) == "## comment ##";
    std::remove(tmpfilename);
    return ok;
}

int main()
{
    std::cout << "Test string formatting and INI file routines\n";
    if(testNumberConversion() && testSplitString() && testIniFile())
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
