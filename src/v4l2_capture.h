#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <linux/videodev2.h>

class V4L2Capture {
public:
    V4L2Capture(const std::string& device, int width, int height);
    ~V4L2Capture();

    V4L2Capture(const V4L2Capture&) = delete;
    V4L2Capture& operator=(const V4L2Capture&) = delete;

    bool init();
    bool startStreaming();
    void stopStreaming();

    // Dequeue a frame buffer. Returns pointer to mmap'd data, sets outSize.
    // Caller must call requeueBuffer() after consuming the frame.
    const uint8_t* dequeueFrame(size_t& outSize);
    void requeueBuffer();

    int width() const { return width_; }
    int height() const { return height_; }
    uint32_t pixelFormat() const { return pixfmt_; }
    size_t frameSize() const { return frameSize_; }

private:
    static constexpr int NUM_BUFFERS = 4;

    struct MmapBuffer {
        void* start = nullptr;
        size_t length = 0;
    };

    std::string device_;
    int fd_ = -1;
    int reqWidth_, reqHeight_;
    int width_ = 0, height_ = 0;
    uint32_t pixfmt_ = 0;
    size_t frameSize_ = 0;
    std::vector<MmapBuffer> buffers_;
    int currentBufIndex_ = -1;
    bool streaming_ = false;
};
