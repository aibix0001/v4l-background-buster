#include "v4l2_capture.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do { r = ioctl(fd, request, arg); } while (r == -1 && errno == EINTR);
    return r;
}

V4L2Capture::V4L2Capture(const std::string& device, int width, int height)
    : device_(device), reqWidth_(width), reqHeight_(height) {}

V4L2Capture::~V4L2Capture() {
    stopStreaming();
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED)
            munmap(buf.start, buf.length);
    }
    if (fd_ >= 0) close(fd_);
}

bool V4L2Capture::init() {
    fd_ = open(device_.c_str(), O_RDWR);
    if (fd_ < 0) {
        fprintf(stderr, "Cannot open %s: %s\n", device_.c_str(), strerror(errno));
        return false;
    }

    // Query capabilities
    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        return false;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is not a capture device\n", device_.c_str());
        return false;
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming\n", device_.c_str());
        return false;
    }

    // Set format — try YUYV first, then MJPEG, then NV12
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = reqWidth_;
    fmt.fmt.pix.height = reqHeight_;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    uint32_t tryFormats[] = { V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_NV12 };
    bool formatSet = false;
    for (auto pf : tryFormats) {
        fmt.fmt.pix.pixelformat = pf;
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == 0) {
            formatSet = true;
            break;
        }
    }
    if (!formatSet) {
        // Fall back to whatever the driver gives us
        fmt.fmt.pix.pixelformat = 0;
        if (xioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) {
            fprintf(stderr, "Cannot get/set format: %s\n", strerror(errno));
            return false;
        }
    }

    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    pixfmt_ = fmt.fmt.pix.pixelformat;
    frameSize_ = fmt.fmt.pix.sizeimage;

    char fourcc[5] = {};
    memcpy(fourcc, &pixfmt_, 4);
    fprintf(stderr, "Capture: %dx%d, format=%s, frameSize=%zu\n",
            width_, height_, fourcc, frameSize_);

    // Request MMAP buffers
    v4l2_requestbuffers req{};
    req.count = NUM_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        fprintf(stderr, "VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        return false;
    }

    buffers_.resize(req.count);
    for (unsigned i = 0; i < req.count; i++) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            return false;
        }
        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s\n", strerror(errno));
            return false;
        }
    }

    return true;
}

bool V4L2Capture::startStreaming() {
    // Queue all buffers
    for (unsigned i = 0; i < buffers_.size(); i++) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QBUF failed: %s\n", strerror(errno));
            return false;
        }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "VIDIOC_STREAMON failed: %s\n", strerror(errno));
        return false;
    }
    streaming_ = true;
    return true;
}

void V4L2Capture::stopStreaming() {
    if (!streaming_) return;
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;
}

const uint8_t* V4L2Capture::dequeueFrame(size_t& outSize) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        fprintf(stderr, "VIDIOC_DQBUF failed: %s\n", strerror(errno));
        outSize = 0;
        return nullptr;
    }
    currentBufIndex_ = buf.index;
    outSize = buf.bytesused;
    return static_cast<const uint8_t*>(buffers_[buf.index].start);
}

void V4L2Capture::requeueBuffer() {
    if (currentBufIndex_ < 0) return;
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = currentBufIndex_;
    xioctl(fd_, VIDIOC_QBUF, &buf);
    currentBufIndex_ = -1;
}
