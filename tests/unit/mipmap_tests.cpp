// mipmap_tests.cpp
#include "../../src/TheRestOfYourLife/rtweekend.h"
#include "../../src/shared/mipmap.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

static std::vector<color> solid_img(int w,int h,color c){return std::vector<color>(w*h,c);}
static std::vector<color> checker_img(int w,int h){std::vector<color> p(w*h);for(int y=0;y<h;++y)for(int x=0;x<w;++x)p[y*w+x]=((x+y)%2==0)?color(0,0,0):color(1,1,1);return p;}

TEST(MipMap,LevelCountPowerOfTwo){mipmap m(solid_img(8,8,color(1,0,0)),8,8);EXPECT_EQ(m.levels(),4);}
