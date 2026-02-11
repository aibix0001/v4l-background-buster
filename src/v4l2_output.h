#pragma once
#include <cstdint>
#include <string>

class V4L2Output {
public:
    V4L2Output(const std::string& device, int width, int height, uint32_t pixfmt);
    ~V4L2Output();

    V4L2Output(const V4L2Output&) = delete;
    V4L2Output& operator=(const V4L2Output&) = delete;

    bool init();
    bool writeFrame(const uint8_t* data, size_t size);

private:
    std::string device_;
    int fd_ = -1;
    int width_, height_;
    uint32_t pixfmt_;
};
