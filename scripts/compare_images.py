import sys
import struct

def read_ppm(filename):
    """Read a PPM file (P3 ASCII or P6 binary) and return width, height, and pixel data as bytes."""
    with open(filename, 'rb') as f:
        # Read header
        magic = f.readline().strip()

        # Skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        # Parse dimensions
        width, height = map(int, line.split())

        # Parse max value
        maxval = int(f.readline().strip())

        if magic == b'P6':
            # Binary format - read pixel data directly
            pixel_data = f.read()
        elif magic == b'P3':
            # ASCII format - parse integers and convert to bytes
            remaining = f.read().decode('ascii')
            values = [int(x) for x in remaining.split()]
            pixel_data = bytes(values)
        else:
            raise ValueError(f"Unknown PPM format: {magic}")

        return width, height, maxval, pixel_data

def compare_images(file1, file2):
    """Compare two PPM images pixel by pixel."""
    w1, h1, max1, data1 = read_ppm(file1)
    w2, h2, max2, data2 = read_ppm(file2)

    print(f"Image 1: {w1}x{h1}, max={max1}, size={len(data1)} bytes")
    print(f"Image 2: {w2}x{h2}, max={max2}, size={len(data2)} bytes")

    if (w1, h1) != (w2, h2):
        print("ERROR: Image dimensions differ!")
        return False

    if len(data1) != len(data2):
        print("ERROR: Pixel data sizes differ!")
        return False

    # Sample a few pixels to compare
    print("\nSampling pixels from different regions:")
    sample_coords = [(0,0), (100,0), (0,100), (100,100), (50,50), (150,150)]
    for x, y in sample_coords:
        if x < w1 and y < h1:
            idx = (y * w1 + x) * 3
            r1, g1, b1 = data1[idx], data1[idx+1], data1[idx+2]
            r2, g2, b2 = data2[idx], data2[idx+1], data2[idx+2]
            match = "✓" if (r1,g1,b1) == (r2,g2,b2) else "✗"
            print(f"  {match} ({x},{y}): CPU=({r1},{g1},{b1}) GPU=({r2},{g2},{b2})")

    # Compare pixel by pixel
    differences = 0
    max_diff = 0
    total_diff = 0
    first_diffs = []

    for i in range(0, len(data1), 3):
        r1, g1, b1 = data1[i], data1[i+1], data1[i+2]
        r2, g2, b2 = data2[i], data2[i+1], data2[i+2]

        if (r1, g1, b1) != (r2, g2, b2):
            differences += 1
            pixel_diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            max_diff = max(max_diff, pixel_diff)
            total_diff += pixel_diff

            # Store first few differences
            if len(first_diffs) < 10:
                pixel_idx = i // 3
                x = pixel_idx % w1
                y = pixel_idx // w1
                first_diffs.append((x, y, r1, g1, b1, r2, g2, b2, pixel_diff))

    total_pixels = w1 * h1

    print(f"\n=== COMPARISON RESULT ===")
    print(f"Total pixels: {total_pixels}")
    print(f"Different pixels: {differences} ({100.0 * differences / total_pixels:.2f}%)")

    if differences > 0:
        print(f"Maximum per-pixel diff: {max_diff}")
        print(f"Average diff per different pixel: {total_diff / differences:.2f}")
        print(f"\nFirst {len(first_diffs)} differences:")
        for x, y, r1, g1, b1, r2, g2, b2, diff in first_diffs:
            print(f"  Pixel ({x},{y}): CPU=({r1},{g1},{b1}) GPU=({r2},{g2},{b2}) [diff={diff}]")
        print(f"\nIMAGES ARE DIFFERENT!")
        return False
    else:
        print(f"\n✓ IMAGES ARE IDENTICAL!")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1.ppm> <image2.ppm>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    result = compare_images(file1, file2)
    sys.exit(0 if result else 1)
