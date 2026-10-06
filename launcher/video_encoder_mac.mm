// video_encoder_mac.mm - see video_encoder_mac.h.
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <cstdio>
#include <unistd.h>

#include "video_encoder_mac.h"

namespace {

CGImageRef load_png(NSString* path) {
    NSURL* url = [NSURL fileURLWithPath:path];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!src) return nullptr;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    return img;  // +1, caller releases
}

// Draws `img` into a new BGRA pixel buffer of exactly width x height (a 1-pixel crop/stretch for odd sizes).
CVPixelBufferRef make_pixel_buffer(CVPixelBufferPoolRef pool, CGImageRef img, size_t width, size_t height) {
    CVPixelBufferRef pb = nullptr;
    if (CVPixelBufferPoolCreatePixelBuffer(nullptr, pool, &pb) != kCVReturnSuccess || !pb) return nullptr;
    CVPixelBufferLockBaseAddress(pb, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(CVPixelBufferGetBaseAddress(pb), width, height, 8,
                                             CVPixelBufferGetBytesPerRow(pb), cs,
                                             kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    if (ctx) {
        CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), img);
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(cs);
    CVPixelBufferUnlockBaseAddress(pb, 0);
    if (!ctx) { CVPixelBufferRelease(pb); return nullptr; }
    return pb;  // +1, caller releases
}

}  // namespace

bool encode_png_sequence_to_mp4_avfoundation(const std::string& framesDir, int frameCount, int fps,
                                             const std::string& outPath, std::string& error) {
    @autoreleasepool {
        if (frameCount <= 0 || fps <= 0) { error = "no frames to encode"; return false; }
        NSString* dir = [NSString stringWithUTF8String:framesDir.c_str()];
        auto framePath = [&](int i) { return [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"enc_%04d.png", i]]; };

        CGImageRef first = load_png(framePath(0));
        if (!first) { error = "could not read " + std::string(framePath(0).UTF8String); return false; }
        const size_t width = std::max<size_t>(2, CGImageGetWidth(first) & ~(size_t)1);
        const size_t height = std::max<size_t>(2, CGImageGetHeight(first) & ~(size_t)1);
        CGImageRelease(first);

        NSString* out = [NSString stringWithUTF8String:outPath.c_str()];
        [[NSFileManager defaultManager] removeItemAtPath:out error:nil];   // AVAssetWriter refuses to overwrite
        NSError* nserr = nil;
        AVAssetWriter* writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:out] fileType:AVFileTypeMPEG4 error:&nserr];
        if (!writer) { error = nserr ? nserr.localizedDescription.UTF8String : "could not create the MP4 writer"; return false; }
        writer.shouldOptimizeForNetworkUse = YES;   // moov atom first ("faststart"): reliable seeking in the GUI's preview

        const double bitsPerPixel = 0.12;   // high quality for synthetic renders; comparable to x264 CRF ~20
        const long bitrate = std::max<long>(2000000, (long)(width * height * fps * bitsPerPixel));
        NSDictionary* settings = @{
            AVVideoCodecKey: AVVideoCodecTypeH264,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height),
            AVVideoColorPropertiesKey: @{
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            },
            AVVideoCompressionPropertiesKey: @{
                AVVideoAverageBitRateKey: @(bitrate),
                AVVideoMaxKeyFrameIntervalKey: @(fps),                // one keyframe per second, like the ffmpeg path's -g
                AVVideoExpectedSourceFrameRateKey: @(fps),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            },
        };
        AVAssetWriterInput* input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:settings];
        input.expectsMediaDataInRealTime = NO;
        AVAssetWriterInputPixelBufferAdaptor* adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:@{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferWidthKey: @(width),
                (id)kCVPixelBufferHeightKey: @(height),
            }];
        if (![writer canAddInput:input]) { error = "the MP4 writer rejected the video settings"; return false; }
        [writer addInput:input];
        if (![writer startWriting]) {
            error = writer.error ? writer.error.localizedDescription.UTF8String : "could not start writing";
            return false;
        }
        [writer startSessionAtSourceTime:kCMTimeZero];

        for (int i = 0; i < frameCount; ++i) {
            @autoreleasepool {
                int waitedMs = 0;
                while (!input.isReadyForMoreMediaData) {
                    if (writer.status == AVAssetWriterStatusFailed || waitedMs > 60000) {
                        error = writer.error ? writer.error.localizedDescription.UTF8String : "the encoder stalled";
                        [writer cancelWriting];
                        return false;
                    }
                    usleep(2000);
                    waitedMs += 2;
                }
                CGImageRef img = load_png(framePath(i));
                if (!img) { error = "could not read " + std::string(framePath(i).UTF8String); [writer cancelWriting]; return false; }
                CVPixelBufferRef pb = make_pixel_buffer(adaptor.pixelBufferPool, img, width, height);
                CGImageRelease(img);
                if (!pb) { error = "could not allocate a video frame"; [writer cancelWriting]; return false; }
                const BOOL ok = [adaptor appendPixelBuffer:pb withPresentationTime:CMTimeMake(i, fps)];
                CVPixelBufferRelease(pb);
                if (!ok) {
                    error = writer.error ? writer.error.localizedDescription.UTF8String : "the encoder rejected a frame";
                    [writer cancelWriting];
                    return false;
                }
            }
        }

        [input markAsFinished];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (writer.status != AVAssetWriterStatusCompleted) {
            error = writer.error ? writer.error.localizedDescription.UTF8String : "the MP4 could not be finalized";
            return false;
        }
        return true;
    }
}
