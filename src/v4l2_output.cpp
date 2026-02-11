#include "v4l2_output.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/videodev2.h>

static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do { r = ioctl(fd, request, arg); } while (r == -1 && errno == EINTR);
    return r;
}

V4L2Output::V4L2Output(const std::string& device, int width, int height, uint32_t pixfmt)
    : device_(device), width_(width), height_(height), pixfmt_(pixfmt) {}

V4L2Output::~V4L2Output() {
    if (fd_ >= 0) close(fd_);
}

bool V4L2Output::init() {
    fd_ = open(device_.c_str(), O_RDWR);
    if (fd_ < 0) {
        fprintf(stderr, "Cannot open output %s: %s\n", device_.c_str(), strerror(errno));
        return false;
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = pixfmt_;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    // #7: Set colorspace metadata for correct YUV interpretation
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SMPTE170M;
    fmt.fmt.pix.ycbcr_enc = V4L2_YCBCR_ENC_601;
    fmt.fmt.pix.quantization = V4L2_QUANTIZATION_LIM_RANGE;

    // F17: Calculate sizeimage with overflow-safe cast
    if (pixfmt_ == V4L2_PIX_FMT_YUYV)
        fmt.fmt.pix.sizeimage = static_cast<uint32_t>(width_) * height_ * 2;
    else if (pixfmt_ == V4L2_PIX_FMT_BGR32 || pixfmt_ == V4L2_PIX_FMT_RGB32)
        fmt.fmt.pix.sizeimage = static_cast<uint32_t>(width_) * height_ * 4;
    else
        fmt.fmt.pix.sizeimage = static_cast<uint32_t>(width_) * height_ * 2;

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "VIDIOC_S_FMT on output failed: %s\n", strerror(errno));
        return false;
    }

    fprintf(stderr, "Output: %dx%d to %s\n", width_, height_, device_.c_str());
    return true;
}

bool V4L2Output::writeFrame(const uint8_t* data, size_t size) {
    ssize_t written = write(fd_, data, size);
    if (written < 0) {
        fprintf(stderr, "write to output failed: %s\n", strerror(errno));
        return false;
    }
    // F13: Check for partial write
    if (static_cast<size_t>(written) != size) {
        fprintf(stderr, "partial write to output: %zd of %zu bytes\n", written, size);
        return false;
    }
    return true;
}
