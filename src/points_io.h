#pragma once

#include "defs.h"

PointSet ReadPoints(const std::string& path, int64_t size = -1);

// Memory-map the file instead of reading it into RAM.
// The OS pages data in on demand, so 500M+ vectors no longer cause OOM at open.
// Only supported for .fbin files (raw float32).
PointSet ReadPointsMmap(const std::string& path);

void WritePoints(PointSet& points, const std::string& path);

std::vector<NNVec> ReadGroundTruth(const std::string& path);

void WriteGroundTruth(const std::string& path, const std::vector<NNVec>& ground_truth);
