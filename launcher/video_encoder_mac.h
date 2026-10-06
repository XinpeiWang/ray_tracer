// video_encoder_mac.h
//
// macOS-only: assemble a numbered PNG sequence into an H.264 MP4 with AVFoundation, which ships with every Mac, so
// Video -> MP4 works without installing ffmpeg. (A GUI app also gets a minimal PATH from launchd - /usr/bin:/bin:... -
// so even a Homebrew ffmpeg in /opt/homebrew/bin is usually NOT found by the render subprocess the GUI starts.)
#pragma once

#include <string>

// Reads <framesDir>/enc_0000.png ... enc_<frameCount-1>.png (the sequence launcher/main.cpp's BackgroundPngConverter
// writes) and writes `outPath` as MP4 (H.264, BT.709, one keyframe per second, moov atom up front so players can seek).
// Odd image sizes are rounded down to even (H.264 4:2:0 needs it). Returns false and fills `error` on failure.
bool encode_png_sequence_to_mp4_avfoundation(const std::string& framesDir, int frameCount, int fps,
                                             const std::string& outPath, std::string& error);
