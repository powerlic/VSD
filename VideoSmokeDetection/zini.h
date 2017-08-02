///////////////////////////////////////////////////////////////
// Name: ZIni
// Version: 0.1
// Author: zhaobinjie
// Email: zhaobinjie1984@163.com
// Date: 2010-04-29
// 
// Description:
//     This class provides static functions for writing and
//     reading INI files.
///////////////////////////////////////////////////////////////

#ifndef ZINI_H
#define ZINI_H

#include <string>

class ZIni
{
public:
    ZIni() {}
    virtual ~ZIni() {}
public:
    // return if succeed or not
    static bool writeString(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        std::string strValue, // in. value to be write
        std::string strFileName // in. INI file name
        );
    // return if succeed or not
    static bool writeInt(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        int iValue, // in. value to be write
        std::string strFileName // in. INI file name
        );
    // return if succeed or not
    static bool writeDouble(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        double fValue, // in. value to be write
        std::string strFileName // in. INI file name
        );
    // return the got string
    static std::string readString(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        std::string strDefault, // in. defaut value if read failed
        std::string strFileName // in. INI file name
        );
    // return the got int
    static int readInt(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        int iDefault, // in. defaut value if read failed
        std::string strFileName // in. INI file name
        );
    // return the got double
    static double readDouble(
        std::string strSectName, // in. section name
        std::string strKeyName, // in. key name
        double fDefault, // in. defaut value if read failed
        std::string strFileName // in. INI file name
        );
private:
    static std::string eatSpace(std::string &strInput);
};

#endif //ZINI_H

