#include "../src/external/image_writer.h"
#include <iostream>

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " <input.ppm> <output.png>" << std::endl;
		return 1;
	}

	if (convert_ppm_to_png(argv[1], argv[2])) {
		std::cout << "Successfully converted " << argv[1] << " to " << argv[2] << std::endl;
		return 0;
	} else {
		std::cerr << "Conversion failed!" << std::endl;
		return 1;
	}
}
