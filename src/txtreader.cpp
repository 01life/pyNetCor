#include "txtreader.h"

#include <sstream>
#include <stdexcept>

TxtReader::TxtReader(std::string txtFile, bool setHeader, bool setIndex) {
    fileName = txtFile;
    mGzipInputBufferSize = GZIP_IN_BUF_SIZE;
    mGzipInputBuffer = new char[mGzipInputBufferSize];
    this->setHeader = setHeader;
    this->setIndex = setIndex;

    if (isGzipFile()) {
        mGzipped = true;
        gzipFileStream = gzopen(fileName.c_str(), "rb");
        if (gzipFileStream == NULL) {
            throw std::invalid_argument("Error: cannot open file " + fileName);
        }
    } else {
        mGzipped = false;
        fileStream.open(fileName);
        if (!fileStream.is_open()) {
            throw std::invalid_argument("Error: cannot open file " + fileName);
        }
    }

    if (setHeader) {
        readHeader();
    }
}

TxtReader::~TxtReader() {
    if (mGzipped) {
        if (gzipFileStream != nullptr) {
            gzclose(gzipFileStream);
        }
    } else {
        if (fileStream.is_open()) {
            fileStream.close();
        }
    }
    delete[] mGzipInputBuffer;
}

bool TxtReader::hasNext() {
    if (mGzipped) {
        return !gzeof(gzipFileStream);
    } else {
        return !fileStream.eof();
    }
}

std::vector<double> TxtReader::readNext() {
    getline();
    std::istringstream ss(line);
    double item;
    std::vector<double> row;

    std::string index;
    if (setIndex && ss >> index) {
        currentIndex = index;
    }

    while (ss >> item) {
        row.push_back(item);
    }
    line.clear();
    return row;
}

std::vector<std::vector<double>> TxtReader::readAll() {
    std::vector<std::vector<double>> data;
    while (hasNext()) {
        std::vector<double> row = readNext();
        // skip empty rows
        if (row.empty()) {
            continue;
        }
        data.push_back(row);
    }
    return data;
}

void TxtReader::readHeader() {
    getline();
    std::istringstream ss(line);
    std::string item;
    while (ss >> item) {
        header.push_back(item);
    }
    line.clear();
}

void TxtReader::getline() {
    if (mGzipped) {
        while (gzgets(gzipFileStream, mGzipInputBuffer, mGzipInputBufferSize) != NULL) {
            line += mGzipInputBuffer;

            if (line.back() == '\n') {
                line.pop_back();
                break;
            }
        }
    }
    else {
        std::getline(fileStream, line);
    }
}

bool TxtReader::isGzipFile() {
    std::ifstream fs(fileName, std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }

    char byte1, byte2;
    fs >> byte1 >> byte2;
    fs.close();
    return (byte1 == char(0x1F)) && (byte2 == char(0x8B)); // Check magic number of gzip
}