//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"

#include "camera.h"
#include "cornell_box_scene.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"


int main(int argc, char** argv) {
    // Build the standard Cornell box scene
    hittable_list world = build_cornell_box_scene();

    // Build light sources for importance sampling
    hittable_list lights = build_cornell_box_lights();

    camera cam;

    cam.aspect_ratio      = 1.0;
    // Default settings
    cam.image_width       = 600;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    // Allow overriding via command-line: <image_width> <samples_per_pixel> <max_depth>
    if (argc >= 4) {
        try {
            int w = std::stoi(argv[1]);
            int spp = std::stoi(argv[2]);
            int md = std::stoi(argv[3]);
            if (w > 0) cam.image_width = w;
            if (spp > 0) cam.samples_per_pixel = spp;
            if (md > 0) cam.max_depth = md;
            std::clog << "Using command-line settings: width=" << cam.image_width
                      << " spp=" << cam.samples_per_pixel << " max_depth=" << cam.max_depth << std::endl;
        } catch (const std::exception& e) {
            std::clog << "Invalid command-line arguments, using defaults: " << e.what() << std::endl;
        }
    }
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0;

    cam.render(world, lights);
    return 0;
}
