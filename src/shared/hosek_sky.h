#pragma once
// hosek_sky.h -- the Hosek-Wilkie analytic sky-dome radiance model (XYZ colour space version): the clear-sky colour and brightness in every direction for a given
// sun height, atmospheric turbidity (haze) and ground albedo. It is the model Blender's Cycles offers as its "Hosek-Wilkie" sky.
//
// Lukas Hosek and Alexander Wilkie, "An Analytic Model for Full Spectral Sky-Dome Radiance", ACM Transactions on Graphics (SIGGRAPH 2012). The coefficient tables
// (src/external/hosek_sky_data.h) are the authors' own, as shipped in Cycles; this file is a compact re-write of the evaluation code of their sample implementation (version
// 1.4a, XYZ model only, no solar disc: the sun is handled in scene_sky.h), keeping its maths.
//
// SPDX-FileCopyrightText: 2012-2013 Lukas Hosek and Alexander Wilkie. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: redistributions
// of source code must retain the above copyright notice, this list of conditions and the following disclaimer; redistributions in binary form must reproduce them
// in the documentation and/or other materials provided with the distribution; the names of the copyright holders and contributors may not be used to endorse or
// promote products derived from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND (see
// licenses/BSD-3-Clause-Hosek-Wilkie.txt for the full text).
//
// std-only.

#include "../external/hosek_sky_data.h"

#include <algorithm>
#include <cmath>

namespace hosek_sky {

class Model {
public:
	// turbidity in [1, 10] (2 is a very clear day, 3 ordinary, 6+ hazy), albedo in [0, 1] (the ground the sky light bounces off), the sun's height above the horizon in
	// radians in [0, pi/2].
	Model(double turbidity, double albedo, double solarElevation) {
		turbidity = std::min(10.0, std::max(1.0, turbidity));
		albedo = std::min(1.0, std::max(0.0, albedo));
		solarElevation = std::min(kHalfPi, std::max(0.0, solarElevation));
		for (int c = 0; c < 3; ++c) {
			cook(data::datasetsXYZ[c], 9, turbidity, albedo, solarElevation, m_config[c]);
			cook(data::datasetsXYZRad[c], 1, turbidity, albedo, solarElevation, &m_radiance[c]);
		}
	}

	// CIE XYZ radiance seen in a direction `theta` radians from the zenith and `gamma` radians from the sun (theta in [0, pi/2]: the model has no ground). In
	// W / (m^2 sr nm) folded with the colour matching functions, so Y * 683 is luminance in cd/m^2.
	void xyz(double theta, double gamma, double out[3]) const {
		theta = std::min(kHalfPi - 1e-4, std::max(0.0, theta));
		for (int c = 0; c < 3; ++c) out[c] = radiance(m_config[c], theta, gamma) * m_radiance[c];
	}

private:
	static constexpr double kHalfPi = 1.5707963267948966;

	// The coefficients for this sun height: a degree-5 Bezier in the cube root of the (normalised) height, blended over the two neighbouring turbidities and the
	// two albedos (0 and 1) the tables hold. `params` values per control point: 9 for the sky configuration, 1 for the radiance scale.
	static void cook(const double* dataset, int params, double turbidity, double albedo, double elevation, double* out) {
		const int t = static_cast<int>(turbidity);
		const double tr = turbidity - t;
		const double s = std::pow(elevation / kHalfPi, 1.0 / 3.0), r = 1.0 - s;
		const double w[6] = {std::pow(r, 5), 5 * std::pow(r, 4) * s, 10 * std::pow(r, 3) * s * s, 10 * r * r * std::pow(s, 3), 5 * r * std::pow(s, 4), std::pow(s, 5)};
		const int block = params * 6;
		for (int i = 0; i < params; ++i) out[i] = 0.0;
		auto add = [&](int offset, double weight) {
			const double* m = dataset + offset;
			for (int i = 0; i < params; ++i) {
				double sum = 0.0;
				for (int j = 0; j < 6; ++j) sum += w[j] * m[i + params * j];
				out[i] += weight * sum;
			}
		};
		add(block * (t - 1), (1.0 - albedo) * (1.0 - tr));
		add(block * 10 + block * (t - 1), albedo * (1.0 - tr));
		if (t != 10) {
			add(block * t, (1.0 - albedo) * tr);
			add(block * 10 + block * t, albedo * tr);
		}
	}

	static double radiance(const double* c, double theta, double gamma) {
		const double cg = std::cos(gamma);
		const double expM = std::exp(c[4] * gamma);
		const double rayM = cg * cg;
		const double mieM = (1.0 + cg * cg) / std::pow(1.0 + c[8] * c[8] - 2.0 * c[8] * cg, 1.5);
		const double zenith = std::sqrt(std::cos(theta));
		return (1.0 + c[0] * std::exp(c[1] / (std::cos(theta) + 0.01))) * (c[2] + c[3] * expM + c[5] * rayM + c[6] * mieM + c[7] * zenith);
	}

	double m_config[3][9];
	double m_radiance[3];
};

}  // namespace hosek_sky
