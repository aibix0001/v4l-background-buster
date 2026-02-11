#include "v4l2_capture.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
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
    fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK);
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
            // #4: Verify driver actually accepted the requested format
            if (fmt.fmt.pix.pixelformat != pf) {
                char req4[5] = {}, got4[5] = {};
                memcpy(req4, &pf, 4);
                memcpy(got4, &fmt.fmt.pix.pixelformat, 4);
                fprintf(stderr, "Driver substituted format %s instead of %s, trying next\n",
                        got4, req4);
                continue;
            }
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

    // #5: Negotiate frame rate (request 60fps)
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 60;
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) == 0) {
        auto& tf = parm.parm.capture.timeperframe;
        float fps = (tf.numerator > 0) ? static_cast<float>(tf.denominator) / tf.numerator : 0;
        fprintf(stderr, "Frame rate: %.1f fps (requested 60)\n", fps);
    } else {
        fprintf(stderr, "Frame rate negotiation not supported by driver\n");
    }

    // F8: Reject odd widths (YUYV and other packed formats require even width)
    if (width_ % 2 != 0) {
        fprintf(stderr, "Negotiated width %d is odd — not supported (must be even)\n", width_);
        return false;
    }

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
    // #3: poll() before DQBUF to avoid infinite hang on USB disconnect
    struct pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 2000);  // 2-second timeout
    if (ret <= 0) {
        if (ret == 0)
            fprintf(stderr, "poll() timeout — camera not responding\n");
        else
            fprintf(stderr, "poll() failed: %s\n", strerror(errno));
        outSize = 0;
        return nullptr;
    }

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // #15: Retry on EAGAIN and transient EIO
    for (int attempt = 0; attempt < 3; attempt++) {
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) == 0)
            break;
        if (errno == EAGAIN)
            continue;
        if (errno == EIO && attempt < 2) {
            fprintf(stderr, "VIDIOC_DQBUF EIO (transient USB error), retrying\n");
            continue;
        }
        fprintf(stderr, "VIDIOC_DQBUF failed: %s\n", strerror(errno));
        outSize = 0;
        return nullptr;
    }

    // F3: Validate buf.index before accessing buffers_
    if (buf.index >= buffers_.size()) {
        fprintf(stderr, "VIDIOC_DQBUF returned invalid index %u (have %zu buffers)\n",
                buf.index, buffers_.size());
        outSize = 0;
        return nullptr;
    }
    currentBufIndex_ = buf.index;
    outSize = buf.bytesused;
    return static_cast<const uint8_t*>(buffers_[buf.index].start);
}

const uint8_t* V4L2Capture::dequeueLatestFrame(size_t& outSize) {
    // #6: Drain stale frames, keeping only the newest
    const uint8_t* latest = dequeueFrame(outSize);
    if (!latest) return nullptr;

    // Attempt additional non-blocking DQBUFs to drain stale frames
    for (;;) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0)
            break;  // EAGAIN = no more frames queued

        if (buf.index >= buffers_.size())
            break;

        // Requeue the older frame we were holding
        requeueBuffer();

        // Keep this newer frame
        currentBufIndex_ = buf.index;
        outSize = buf.bytesused;
        latest = static_cast<const uint8_t*>(buffers_[buf.index].start);
    }
    return latest;
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
