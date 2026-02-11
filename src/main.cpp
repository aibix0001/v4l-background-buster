#include "pipeline.h"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>

static volatile sig_atomic_t g_running = 1;

static void signalHandler(int) { g_running = 0; }

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  -i, --input DEVICE        Camera device (default: /dev/video0)\n"
        "  -o, --output DEVICE       v4l2loopback device (default: /dev/video10)\n"
        "  -W, --width N             Frame width (default: 1920)\n"
        "  -H, --height N            Frame height (default: 1080)\n"
        "  -m, --model PATH          ONNX model path (default: auto from resolution)\n"
        "  -e, --engine PATH         TensorRT plan cache (default: auto from resolution)\n"
        "  -d, --downsample RATIO    RVM downsample ratio (default: 0.25)\n"
        "  -b, --background MODE     green|color (default: green)\n"
        "  -c, --color R,G,B         Background color for 'color' mode\n"
        "  -s, --smooth FACTOR       Alpha temporal smoothing (0.0-1.0, default: 1.0=off)\n"
        "      --no-fp16             Disable FP16 (use FP32)\n"
        "      --benchmark           Print per-frame timing stats\n"
        "  -h, --help                Show this help\n",
        prog);
}

int main(int argc, char* argv[]) {
    PipelineConfig cfg;

    static struct option longOpts[] = {
        {"input",      required_argument, nullptr, 'i'},
        {"output",     required_argument, nullptr, 'o'},
        {"width",      required_argument, nullptr, 'W'},
        {"height",     required_argument, nullptr, 'H'},
        {"model",      required_argument, nullptr, 'm'},
        {"engine",     required_argument, nullptr, 'e'},
        {"downsample", required_argument, nullptr, 'd'},
        {"background", required_argument, nullptr, 'b'},
        {"color",      required_argument, nullptr, 'c'},
        {"smooth",     required_argument, nullptr, 's'},
        {"no-fp16",    no_argument,       nullptr, 1},
        {"benchmark",  no_argument,       nullptr, 2},
        {"help",       no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:W:H:m:e:d:b:c:s:h", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'i': cfg.inputDevice = optarg; break;
            case 'o': cfg.outputDevice = optarg; break;
            case 'W': cfg.width = atoi(optarg); break;
            case 'H': cfg.height = atoi(optarg); break;
            case 'm': cfg.onnxPath = optarg; break;
            case 'e': cfg.planPath = optarg; break;
            case 'd': cfg.downsampleRatio = atof(optarg); break;
            case 'b':
                if (strcmp(optarg, "green") == 0) {
                    cfg.bgR = 0; cfg.bgG = 177; cfg.bgB = 64;
                }
                break;
            case 'c':
                if (sscanf(optarg, "%hhu,%hhu,%hhu", &cfg.bgR, &cfg.bgG, &cfg.bgB) != 3) {
                    fprintf(stderr, "Error: --color requires R,G,B format (e.g. 0,177,64)\n");
                    return 1;
                }
                break;
            case 's': cfg.alphaSmoothing = atof(optarg); break;
            case 1: cfg.fp16 = false; break;
            case 2: cfg.benchmark = true; break;
            case 'h':
            default:
                printUsage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    // F6: Validate width, height, downsample ratio
    if (cfg.width < 1 || cfg.width > 8192) {
        fprintf(stderr, "Error: width must be in range [1, 8192], got %d\n", cfg.width);
        return 1;
    }
    if (cfg.height < 1 || cfg.height > 8192 || cfg.height % 2 != 0) {
        fprintf(stderr, "Error: height must be even and in range [2, 8192], got %d\n", cfg.height);
        return 1;
    }
    if (cfg.downsampleRatio <= 0.0f || cfg.downsampleRatio > 1.0f) {
        fprintf(stderr, "Error: downsample ratio must be in (0.0, 1.0], got %f\n",
                cfg.downsampleRatio);
        return 1;
    }
    if (cfg.alphaSmoothing <= 0.0f || cfg.alphaSmoothing > 1.0f) {
        fprintf(stderr, "Error: smooth factor must be in (0.0, 1.0], got %f\n",
                cfg.alphaSmoothing);
        return 1;
    }

    // F10: Validate device paths
    auto isValidDevicePath = [](const std::string& path) {
        return path.rfind("/dev/video", 0) == 0 || path.rfind("/dev/v4l", 0) == 0;
    };
    if (!isValidDevicePath(cfg.inputDevice)) {
        fprintf(stderr, "Error: input device must start with /dev/video or /dev/v4l, got '%s'\n",
                cfg.inputDevice.c_str());
        return 1;
    }
    if (!isValidDevicePath(cfg.outputDevice)) {
        fprintf(stderr, "Error: output device must start with /dev/video or /dev/v4l, got '%s'\n",
                cfg.outputDevice.c_str());
        return 1;
    }

    // F11: Reject engine path with ".." segments
    if (cfg.planPath.find("..") != std::string::npos) {
        fprintf(stderr, "Error: engine path must not contain '..' segments, got '%s'\n",
                cfg.planPath.c_str());
        return 1;
    }

    // Register signal handlers for clean shutdown (F12: use sigaction)
    struct sigaction sa{};
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    fprintf(stderr, "v4l-background-buster starting...\n");
    fprintf(stderr, "  Input:  %s\n", cfg.inputDevice.c_str());
    fprintf(stderr, "  Output: %s\n", cfg.outputDevice.c_str());
    fprintf(stderr, "  Resolution: %dx%d\n", cfg.width, cfg.height);
    fprintf(stderr, "  Model: %s\n", cfg.onnxPath.c_str());
    fprintf(stderr, "  FP16: %s\n", cfg.fp16 ? "yes" : "no");
    fprintf(stderr, "  Background: RGB(%d,%d,%d)\n", cfg.bgR, cfg.bgG, cfg.bgB);
    if (cfg.alphaSmoothing < 1.0f)
        fprintf(stderr, "  Alpha smoothing: %.2f\n", cfg.alphaSmoothing);

    Pipeline pipeline(cfg);
    if (!pipeline.init()) {
        fprintf(stderr, "Pipeline initialization failed\n");
        return 1;
    }

    fprintf(stderr, "Processing frames... (Ctrl+C to stop)\n");

    long frameCount = 0;
    float totalMs = 0.0f;
    float totalWallMs = 0.0f;

    while (g_running) {
        if (!pipeline.processFrame()) {
            fprintf(stderr, "Frame processing failed at frame %ld\n", frameCount);
            break;
        }
        frameCount++;

        if (cfg.benchmark) {
            float ms = pipeline.lastFrameMs();
            float wallMs = pipeline.lastWallMs();
            totalMs += ms;
            totalWallMs += wallMs;
            if (frameCount % 100 == 0) {
                fprintf(stderr, "Frame %ld: GPU %.2f ms, wall %.2f ms (avg GPU: %.2f ms, wall: %.2f ms, %.1f FPS)\n",
                        frameCount, ms, wallMs, totalMs / frameCount,
                        totalWallMs / frameCount,
                        1000.0f * frameCount / totalWallMs);
            }
        }
    }

    fprintf(stderr, "\nStopping after %ld frames\n", frameCount);
    if (cfg.benchmark && frameCount > 0) {
        fprintf(stderr, "Average GPU: %.2f ms/frame, wall: %.2f ms/frame (%.1f FPS)\n",
                totalMs / frameCount, totalWallMs / frameCount,
                1000.0f * frameCount / totalWallMs);
    }

    return 0;
}
