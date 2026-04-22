#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>

int main() {
    int W = 512;
    int H = 512;
    std::vector<unsigned char> image(W * H * 3, 255.);
    stbi_write_png("image.png", W, H, 3, &image[0], 0);
}