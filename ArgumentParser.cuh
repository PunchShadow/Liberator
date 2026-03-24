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
    ArgumentParser(int argc, char **argv, bool canHaveSource);

    bool Parse();

    string GenerateHelpString();

};