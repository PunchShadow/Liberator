#pragma once
#include <string>
#include <iostream>
using namespace std;
class ArgumentParser {
private:

public:
    int argc;
    char **argv;

    bool canHaveSource;

    bool hasInput;
    bool hasSourceNode;
    string input;
    float adviseK = 0.0f;
    int sourceNode;
    int method;
    string algo;
    int model;
    int testTimes;
    double gpuMemoryLimit = 0.0; // GPU memory limit in GB (0 = use actual GPU memory)
    bool verify = false; // Run CPU verification after GPU computation
    string cacheCsv; // Path for per-iter cache density CSV output (empty = disabled)
    string pathCsv;  // Path for per-iter edge-path breakdown CSV output (empty = disabled)
    ArgumentParser(int argc, char **argv, bool canHaveSource);

    bool Parse();

    string GenerateHelpString();

};