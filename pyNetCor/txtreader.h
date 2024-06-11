#ifndef TXT_READER_H
#define TXT_READER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

const size_t GZIP_IN_BUF_SIZE = 1 << 22;
const size_t GZIP_HEADER_SIZE = 10;

class TxtReader {
public:
    explicit TxtReader(std::string txtFile, bool setHeader = true, bool setIndex = false);
    ~TxtReader();

    bool hasNext();
    std::vector<double> readNext();
    std::vector<std::vector<double>> readAll();
    std::string currentIndex;

    inline bool isGzipped() const {
        return mGzipped;
    }
    inline std::vector<std::string> getHeader() {
        return header;
    }

private:
    std::string fileName;
    std::ifstream fileStream;
    gzFile gzipFileStream;
    bool setHeader;
    bool setIndex;
    std::vector<std::string> header;
    bool mGzipped;
    size_t mGzipInputBufferSize;
    char *mGzipInputBuffer;
    std::string line;

    void readHeader();
    void getline();
    bool isGzipFile();
};

#endif // TXT_READER_H
