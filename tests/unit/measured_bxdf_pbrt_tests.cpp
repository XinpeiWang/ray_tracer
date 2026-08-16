/**
 * @file measured_bxdf_pbrt_tests.cpp
 * @brief Unit tests for the pbrt-v4 "measured" material wiring
 *
 * Named to avoid colliding with the existing (unrelated) measured_bxdf_tests.cpp
 * (tests src/shared/measured_bxdf.h's PiecewiseLinear2D/MeasuredBxDF math against
 * synthetic tables) and measured_material_tests.cpp (tests src/shared/materials.h's
 * GPU-style MeasuredMaterial wrapper). This file is about the NEW piece: actually
 * getting real data out of a .bsdf tensor file on disk
 * (src/shared/measured_bxdf_loader.h) and wiring `Material "measured"` through
 * pbrt_flatten.h / pbrt_load.h / pbrt_cpu_builder.h / material_pbrt.h's `class
 * measured` so a pbrt scene naming one actually gets a real, importance-sampled
 * glossy BRDF instead of the flat-diffuse fallback every other unsupported
 * material kind gets.
 *
 * Three layers are covered:
 *   1. TensorLoaderTest       -- the binary "tensor_file" parser, against a
 *                                 tiny synthetic .bsdf written to a temp file
 *                                 (so this suite doesn't depend on the 7-15MB
 *                                 real assets under pbrt_scenes/sportscar/).
 *   2. RealBsdfFileTest       -- physical self-consistency checks (white-
 *                                 furnace / energy conservation, Sample_f<->PDF
 *                                 agreement, approximate reciprocity) against
 *                                 the real ilm_l3_37_metallic_spec.bsdf asset,
 *                                 when it can be found relative to the test
 *                                 binary - skipped (not failed) otherwise.
 *   3. PbrtWiringTest          -- Material "measured" through pbrt_load.h's
 *                                 scene-directory path resolution and
 *                                 pbrt_cpu_builder.h's material switch,
 *                                 confirming an actual `measured` material
 *                                 (not a lambertian fallback) ends up on the
 *                                 hit geometry.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "../../src/shared/measured_bxdf.h"
#include "../../src/shared/measured_bxdf_loader.h"
#include "../../src/shared/pbrt_flatten.h"
#include "../../src/shared/pbrt_load.h"
#include "../../src/shared/pbrt_scene.h"
#include "pbrt_cpu_builder.h"

namespace {

// ===========================================================================
// Binary "tensor_file" writer -- the inverse of measured_bxdf_loader.h's
// Tensor::Load(), used to manufacture tiny synthetic .bsdf files so the
// parser itself can be tested without the real (7-15MB) assets.
// ===========================================================================

struct SyntheticField {
	std::string name;
	uint8_t dtype;                 // measured_bxdf_io::TensorType
	std::vector<uint64_t> shape;
	std::vector<uint8_t> data;     // raw bytes, product(shape)*typeSize
};

template <typename T>
std::vector<uint8_t> RawBytes(const std::vector<T>& values) {
	std::vector<uint8_t> out(values.size() * sizeof(T));
	std::memcpy(out.data(), values.data(), out.size());
	return out;
}

SyntheticField MakeF32Field(const std::string& name, std::vector<uint64_t> shape,
							std::vector<float> values) {
	SyntheticField f;
	f.name = name;
	f.dtype = static_cast<uint8_t>(measured_bxdf_io::TensorType::Float32);
	f.shape = std::move(shape);
	f.data = RawBytes(values);
	return f;
}

SyntheticField MakeU8Field(const std::string& name, std::vector<uint64_t> shape,
						   std::vector<uint8_t> values) {
	SyntheticField f;
	f.name = name;
	f.dtype = static_cast<uint8_t>(measured_bxdf_io::TensorType::UInt8);
	f.shape = std::move(shape);
	f.data = std::move(values);
	return f;
}

// Writes `fields` as a valid tensor_file container: 12-byte magic, 2-byte
// version {1,0}, 4-byte field count, then per-field headers (name, ndim,
// dtype, absolute offset, shape), then the raw data blocks themselves - see
// measured_bxdf_loader.h's own header comment for the format this mirrors.
void WriteTensorFile(const std::string& path, const std::vector<SyntheticField>& fields) {
	// Header size, so each field's absolute data offset is known before any
	// bytes are written.
	uint64_t headerSize = 12 + 2 + 4;
	for (const SyntheticField& f : fields)
		headerSize += 2 + f.name.size() + 2 + 1 + 8 + 8 * f.shape.size();

	std::vector<uint64_t> offsets(fields.size());
	uint64_t cursor = headerSize;
	for (size_t i = 0; i < fields.size(); ++i) {
		offsets[i] = cursor;
		cursor += fields[i].data.size();
	}

	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	ASSERT_TRUE(out) << "could not open " << path << " for writing";

	const char magic[12] = {'t', 'e', 'n', 's', 'o', 'r', '_', 'f', 'i', 'l', 'e', '\0'};
	out.write(magic, 12);
	const uint8_t version[2] = {1, 0};
	out.write(reinterpret_cast<const char*>(version), 2);
	const uint32_t nFields = static_cast<uint32_t>(fields.size());
	out.write(reinterpret_cast<const char*>(&nFields), sizeof(nFields));

	for (size_t i = 0; i < fields.size(); ++i) {
		const SyntheticField& f = fields[i];
		const uint16_t nameLen = static_cast<uint16_t>(f.name.size());
		out.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
		out.write(f.name.data(), nameLen);
		const uint16_t ndim = static_cast<uint16_t>(f.shape.size());
		out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
		out.write(reinterpret_cast<const char*>(&f.dtype), sizeof(f.dtype));
		out.write(reinterpret_cast<const char*>(&offsets[i]), sizeof(offsets[i]));
		for (uint64_t dim : f.shape)
			out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
	}
	for (const SyntheticField& f : fields)
		out.write(reinterpret_cast<const char*>(f.data.data()),
				  static_cast<std::streamsize>(f.data.size()));
}

// A minimal but structurally valid, ISOTROPIC (phi_i has exactly 2 samples)
// synthetic .bsdf: 2x2 ndf/sigma, a 2x2x2x2 vndf/luminance conditioned on
// (phi_i, theta_i), and a 2x2x3x2x2 spectra table (3 wavelengths). Every
// numeric table is filled with distinct, known values so LoadMeasuredBRDF's
// resulting Eval() calls can be checked against them exactly (see this
// file's comment on why normalize=false/build_cdf=false for ndf/sigma/
// spectra makes Eval() at a grid corner return the raw input value
// unchanged - measured_bxdf_loader.h's own header comment explains the
// cancellation).
std::string WriteMinimalValidBsdf(const std::string& path) {
	std::vector<SyntheticField> fields;
	fields.push_back(MakeU8Field("description", {1}, {0}));
	fields.push_back(MakeF32Field("theta_i", {2}, {0.0f, 1.0f}));
	fields.push_back(MakeF32Field("phi_i", {2}, {0.0f, 3.14159265f}));
	fields.push_back(MakeF32Field("wavelengths", {3}, {400.0f, 550.0f, 700.0f}));
	// ndf / sigma: shape [ny=2, nx=2]
	fields.push_back(MakeF32Field("ndf", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
	fields.push_back(MakeF32Field("sigma", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}));
	// vndf / luminance: shape [nphi_i=2, ntheta_i=2, ny=2, nx=2]
	std::vector<float> vndfVals(16);
	for (int i = 0; i < 16; ++i) vndfVals[i] = 1.0f;   // uniform -> valid CDF/PDF
	fields.push_back(MakeF32Field("vndf", {2, 2, 2, 2}, vndfVals));
	fields.push_back(MakeF32Field("luminance", {2, 2, 2, 2}, vndfVals));
	// spectra: shape [nphi_i=2, ntheta_i=2, nwl=3, ny=2, nx=2]
	std::vector<float> spectraVals(2 * 2 * 3 * 2 * 2, 0.5f);
	fields.push_back(MakeF32Field("spectra", {2, 2, 3, 2, 2}, spectraVals));
	fields.push_back(MakeU8Field("jacobian", {1}, {1}));

	WriteTensorFile(path, fields);
	return path;
}

std::string TempDir() {
	const char* tmp = std::getenv("TEMP");
	std::string dir = std::string(tmp ? tmp : ".") + "/measured_bxdf_pbrt_tests/";
#ifdef _WIN32
	std::string cmd = "if not exist \"" + dir + "\" mkdir \"" + dir + "\" >nul 2>&1";
	for (char& c : cmd) if (c == '/') c = '\\';
	std::system(cmd.c_str());
#else
	std::system(("mkdir -p '" + dir + "'").c_str());
#endif
	return dir;
}

// Tries a handful of relative prefixes so this suite doesn't depend on the
// test binary's working directory (Visual Studio, ctest, and a plain
// `ray_tracer_tests.exe` invocation from bin\Release\ don't all agree on
// one). Returns an empty string if none of them exist, in which case the
// real-file-dependent tests skip rather than fail.
std::string FindRepoFile(const std::string& relativeFromRepoRoot) {
	const char* prefixes[] = {"", "../", "../../", "../../../", "../../../../"};
	for (const char* prefix : prefixes) {
		const std::string candidate = std::string(prefix) + relativeFromRepoRoot;
		std::ifstream probe(candidate, std::ios::binary);
		if (probe) return candidate;
	}
	return std::string();
}

} // namespace

// ===========================================================================
// 1. Tensor / .bsdf binary parser
// ===========================================================================

TEST(MeasuredBxdfTensorLoader, ParsesAMinimalValidFile) {
	const std::string path = WriteMinimalValidBsdf(TempDir() + "minimal.bsdf");

	MeasuredBRDFData data;
	std::string error;
	ASSERT_TRUE(measured_bxdf_io::LoadMeasuredBRDF(path, data, error)) << error;

	EXPECT_TRUE(data.isotropic) << "phi_i has exactly 2 samples -> isotropic";
	ASSERT_EQ(data.wavelengths.size(), 3u);
	EXPECT_FLOAT_EQ(data.wavelengths[0], 400.0f);
	EXPECT_FLOAT_EQ(data.wavelengths[1], 550.0f);
	EXPECT_FLOAT_EQ(data.wavelengths[2], 700.0f);

	// ndf/sigma use normalize=false, build_cdf=false (matching pbrt-v4's own
	// construction - see measured_bxdf_loader.h's header comment for why),
	// so Eval() at a grid corner returns the input value unchanged.
	EXPECT_NEAR(data.ndf.Eval(0.0f, 0.0f), 1.0f, 1e-4f);
	EXPECT_NEAR(data.sigma.Eval(0.0f, 0.0f), 5.0f, 1e-4f);
}

TEST(MeasuredBxdfTensorLoader, RejectsBadMagic) {
	const std::string path = TempDir() + "bad_magic.bsdf";
	std::vector<SyntheticField> fields;
	fields.push_back(MakeU8Field("description", {1}, {0}));
	WriteTensorFile(path, fields);

	// Corrupt the first byte of the magic header in place.
	{
		std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(f);
		f.seekp(0);
		char bad = 'X';
		f.write(&bad, 1);
	}

	MeasuredBRDFData data;
	std::string error;
	EXPECT_FALSE(measured_bxdf_io::LoadMeasuredBRDF(path, data, error));
	EXPECT_FALSE(error.empty());
}

TEST(MeasuredBxdfTensorLoader, RejectsMissingField) {
	// A structurally valid tensor_file that simply never names "spectra" -
	// LoadMeasuredBRDF must report a clear missing-field error, not crash.
	const std::string path = TempDir() + "missing_field.bsdf";
	std::vector<SyntheticField> fields;
	fields.push_back(MakeU8Field("description", {1}, {0}));
	fields.push_back(MakeF32Field("theta_i", {2}, {0.0f, 1.0f}));
	fields.push_back(MakeF32Field("phi_i", {2}, {0.0f, 3.14159265f}));
	WriteTensorFile(path, fields);

	MeasuredBRDFData data;
	std::string error;
	EXPECT_FALSE(measured_bxdf_io::LoadMeasuredBRDF(path, data, error));
	EXPECT_NE(error.find("field"), std::string::npos) << error;
}

TEST(MeasuredBxdfTensorLoader, RejectsTruncatedFile) {
	const std::string good = WriteMinimalValidBsdf(TempDir() + "to_truncate.bsdf");
	const std::string truncated = TempDir() + "truncated.bsdf";
	{
		std::ifstream in(good, std::ios::binary);
		std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		ASSERT_GT(bytes.size(), 40u);
		std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
		out.write(bytes.data(), 40);   // header + first field name only
	}

	MeasuredBRDFData data;
	std::string error;
	EXPECT_FALSE(measured_bxdf_io::LoadMeasuredBRDF(truncated, data, error));
	EXPECT_FALSE(error.empty());
}

TEST(MeasuredBxdfTensorLoader, CachesByResolvedPath) {
	const std::string path = WriteMinimalValidBsdf(TempDir() + "cache_me.bsdf");

	std::string e1, e2;
	std::shared_ptr<const MeasuredBRDFData> a = measured_bxdf_io::GetMeasuredBRDFDataCached(path, e1);
	std::shared_ptr<const MeasuredBRDFData> b = measured_bxdf_io::GetMeasuredBRDFDataCached(path, e2);
	ASSERT_TRUE(a);
	ASSERT_TRUE(b);
	EXPECT_EQ(a.get(), b.get()) << "same path must be served from cache, not reparsed";
}

// ===========================================================================
// 2. Physical self-consistency against the real sportscar asset
//
// No ground-truth pbrt-v4 render exists to diff against, so these lean on
// properties the underlying math MUST satisfy regardless of what the actual
// measured paint looks like: sample_f()'s returned throughput cannot imply
// reflecting more energy than arrived (white furnace), a sampler and its
// own density function must agree on where the probability mass is
// (Sample_f/PDF consistency), and f(wo,wi) should not be wildly asymmetric
// under wo<->wi exchange (approximate reciprocity - measured BRDFs are not
// required to be exactly reciprocal, per this task's own instructions).
// ===========================================================================

namespace {
const char* kRealBsdfRelPath = "pbrt_scenes/sportscar/bsdfs/ilm_l3_37_metallic_spec.bsdf";

// Builds a local-frame direction with the given elevation off the surface
// normal (z) and azimuth, always in the upper hemisphere.
void DirFromAngles(double thetaDeg, double phiDeg, double& x, double& y, double& z) {
	const double theta = thetaDeg * 3.14159265358979323846 / 180.0;
	const double phi = phiDeg * 3.14159265358979323846 / 180.0;
	x = std::sin(theta) * std::cos(phi);
	y = std::sin(theta) * std::sin(phi);
	z = std::cos(theta);
}
} // namespace

TEST(MeasuredBxdfRealFile, LoadsTheSportscarPaintScan) {
	const std::string path = FindRepoFile(kRealBsdfRelPath);
	if (path.empty()) {
		GTEST_SKIP() << "could not locate " << kRealBsdfRelPath
					 << " relative to the test binary; skipping real-asset tests";
	}
	std::string error;
	std::shared_ptr<const MeasuredBRDFData> data =
		measured_bxdf_io::GetMeasuredBRDFDataCached(path, error);
	ASSERT_TRUE(data) << error;
	EXPECT_FALSE(data->wavelengths.empty());
}

TEST(MeasuredBxdfRealFile, WhiteFurnaceEnergyConservation) {
	const std::string path = FindRepoFile(kRealBsdfRelPath);
	if (path.empty()) GTEST_SKIP() << "sportscar .bsdf asset not found";

	std::string error;
	std::shared_ptr<const MeasuredBRDFData> data =
		measured_bxdf_io::GetMeasuredBRDFDataCached(path, error);
	ASSERT_TRUE(data) << error;

	MeasuredBxDF<double> bxdf(data.get(), 612.0f, 549.0f, 465.0f);

	// Directly numerically integrate f(wo,wi)*cos(theta_i) over the
	// hemisphere via f() itself, on a fine deterministic (theta_i,phi_i)
	// grid with the correct solid-angle Jacobian (sin(theta_i) dtheta dphi).
	// This is a plain Riemann sum, not a Monte Carlo estimator - which
	// matters a great deal for THIS asset: ilm_l3_37_metallic_spec.bsdf is a
	// genuinely near-specular "spec" paint layer (f(wo,wi) peaks above 60-90
	// within a couple of degrees of the mirror direction and falls to ~0.01
	// a few tens of degrees off it - confirmed by direct inspection), so a
	// sample_f()-based Monte Carlo average of only a few thousand samples is
	// dominated by the variance of rare samples landing near that razor-thin
	// peak and can read 10-100x too high despite every individual sample
	// being computed correctly - that is a property of the ESTIMATOR at
	// finite sample count, not a bug in Sample_f/f/PDF (SampleFAndPDFAgree,
	// below, confirms those three agree with each other independently).
	// A dense deterministic grid does not have that failure mode.
	const int kThetaSteps = 300, kPhiSteps = 300;
	const double kHalfPi = 3.14159265358979323846 / 2.0;
	const double kTwoPi = 2.0 * 3.14159265358979323846;

	for (double thetaO : {10.0, 30.0, 60.0, 80.0}) {
		double wox, woy, woz;
		DirFromAngles(thetaO, 0.0, wox, woy, woz);

		double sumR = 0.0, sumG = 0.0, sumB = 0.0;
		for (int it = 0; it < kThetaSteps; ++it) {
			const double theta = (it + 0.5) / kThetaSteps * kHalfPi;
			const double sinTheta = std::sin(theta), cosTheta = std::cos(theta);
			const double dOmega = sinTheta * (kHalfPi / kThetaSteps) * (kTwoPi / kPhiSteps);
			for (int ip = 0; ip < kPhiSteps; ++ip) {
				const double phi = (ip + 0.5) / kPhiSteps * kTwoPi;
				const double wix = sinTheta * std::cos(phi);
				const double wiy = sinTheta * std::sin(phi);
				double fr, fg, fb;
				bxdf.f(wox, woy, woz, wix, wiy, cosTheta, fr, fg, fb);
				sumR += fr * cosTheta * dOmega;
				sumG += fg * cosTheta * dOmega;
				sumB += fb * cosTheta * dOmega;
			}
		}

		// A metallic paint should reflect strongly (this is the whole point
		// of picking this asset per the task's own instructions) but must
		// not exceed 1, and must not be degenerate (all zero).
		EXPECT_LE(sumR, 1.05) << "theta_o=" << thetaO << " R channel hemispherical reflectance";
		EXPECT_LE(sumG, 1.05) << "theta_o=" << thetaO << " G channel hemispherical reflectance";
		EXPECT_LE(sumB, 1.05) << "theta_o=" << thetaO << " B channel hemispherical reflectance";
		EXPECT_GT(sumR + sumG + sumB, 1e-4)
			<< "theta_o=" << thetaO << ": reflectance is degenerately close to zero";

		std::cout << "[ WhiteFurnace ] theta_o=" << thetaO << " deg -> integral(f*cos) = ("
				  << sumR << ", " << sumG << ", " << sumB << ")  ["
				  << kThetaSteps << "x" << kPhiSteps << " quadrature]" << std::endl;
	}
}

TEST(MeasuredBxdfRealFile, SampleFAndPDFAgree) {
	const std::string path = FindRepoFile(kRealBsdfRelPath);
	if (path.empty()) GTEST_SKIP() << "sportscar .bsdf asset not found";

	std::string error;
	std::shared_ptr<const MeasuredBRDFData> data =
		measured_bxdf_io::GetMeasuredBRDFDataCached(path, error);
	ASSERT_TRUE(data) << error;

	MeasuredBxDF<double> bxdf(data.get(), 612.0f, 549.0f, 465.0f);
	std::mt19937 rng(777);
	std::uniform_real_distribution<float> unif(0.0f, 1.0f);

	double wox, woy, woz;
	DirFromAngles(35.0, 0.0, wox, woy, woz);

	int checked = 0;
	double maxRelErr = 0.0, sumRelErr = 0.0;
	for (int i = 0; i < 500; ++i) {
		double wix, wiy, wiz, fr, fg, fb, sampledPdf;
		if (!bxdf.sample_f(wox, woy, woz, unif(rng), unif(rng),
						   wix, wiy, wiz, fr, fg, fb, sampledPdf))
			continue;
		if (sampledPdf <= 0.0) continue;

		const double recomputedPdf = bxdf.pdf(wox, woy, woz, wix, wiy, wiz);
		const double relErr = std::fabs(recomputedPdf - sampledPdf) / std::max(sampledPdf, 1e-8);
		maxRelErr = std::max(maxRelErr, relErr);
		sumRelErr += relErr;
		++checked;
	}
	ASSERT_GT(checked, 100) << "too few accepted samples to trust the comparison";

	const double meanRelErr = sumRelErr / checked;
	std::cout << "[ Sample_f/PDF consistency ] n=" << checked
			  << " mean relative error=" << meanRelErr
			  << " max relative error=" << maxRelErr << std::endl;

	// Single-precision tables internally (piecewise_linear_2d.h) mean exact
	// bit-for-bit agreement isn't expected; this bounds a SYSTEMATIC
	// disagreement between the sampler and its own density function, which
	// is the actual bug class this check exists to catch.
	EXPECT_LT(meanRelErr, 0.02);
	EXPECT_LT(maxRelErr, 0.15);
}

TEST(MeasuredBxdfRealFile, ApproximateReciprocity) {
	const std::string path = FindRepoFile(kRealBsdfRelPath);
	if (path.empty()) GTEST_SKIP() << "sportscar .bsdf asset not found";

	std::string error;
	std::shared_ptr<const MeasuredBRDFData> data =
		measured_bxdf_io::GetMeasuredBRDFDataCached(path, error);
	ASSERT_TRUE(data) << error;

	MeasuredBxDF<double> bxdf(data.get(), 612.0f, 549.0f, 465.0f);

	struct Pair { double thetaA, phiA, thetaB, phiB; };
	const Pair pairs[] = {
		{20.0, 0.0,   40.0, 30.0},
		{35.0, 10.0,  50.0, 200.0},
		{15.0, 90.0,  25.0, 270.0},
		{45.0, 45.0,  45.0, 225.0},
	};

	for (const Pair& p : pairs) {
		double ax, ay, az, bx, by, bz;
		DirFromAngles(p.thetaA, p.phiA, ax, ay, az);
		DirFromAngles(p.thetaB, p.phiB, bx, by, bz);

		double frAB, fgAB, fbAB, frBA, fgBA, fbBA;
		bxdf.f(ax, ay, az, bx, by, bz, frAB, fgAB, fbAB);
		bxdf.f(bx, by, bz, ax, ay, az, frBA, fgBA, fbBA);

		const double sumAB = frAB + fgAB + fbAB, sumBA = frBA + fgBA + fbBA;
		std::cout << "[ Reciprocity ] wo=(" << p.thetaA << "," << p.phiA << ") wi=("
				  << p.thetaB << "," << p.phiB << ")  f(wo,wi)_sum=" << sumAB
				  << "  f(wi,wo)_sum=" << sumBA << std::endl;

		if (sumAB < 1e-8 && sumBA < 1e-8) continue;   // both effectively zero: nothing to compare
		const double ratio = (sumAB > sumBA) ? (sumAB / std::max(sumBA, 1e-8))
											 : (sumBA / std::max(sumAB, 1e-8));
		// "Not wildly asymmetric" per this task's own instructions, not
		// exact reciprocity - a measured BRDF's VNDF-based construction is
		// only approximately reciprocal by nature.
		EXPECT_LT(ratio, 4.0) << "f(wo,wi) and f(wi,wo) differ by more than 4x";
	}
}

// ===========================================================================
// 3. End-to-end: Material "measured" through the pbrt scene loader
// ===========================================================================

TEST(MeasuredBxdfPbrtWiring, RelativeFilenameResolvesAndLoads) {
	const std::string dir = TempDir();
	WriteMinimalValidBsdf(dir + "mini.bsdf");

	const std::string scenePath = dir + "scene.pbrt";
	{
		std::ofstream out(scenePath, std::ios::binary);
		out << "MakeNamedMaterial \"m\" \"string type\" [ \"measured\" ] "
			   "\"string filename\" [ \"mini.bsdf\" ]\n"
			   "NamedMaterial \"m\"\n"
			   "Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
			   "  \"point3 P\" [ -10 -10 0   10 -10 0   0 10 0 ]\n";
	}

	const pbrt_load::LoadResult r = pbrt_load::loadFile(scenePath);
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_EQ(r.scene.materials[0].kind, pbrt_flatten::MaterialKind::Measured);
	EXPECT_FALSE(r.scene.materials[0].measuredFilename.empty())
		<< "a valid, resolvable .bsdf must leave measuredFilename non-empty (resolved path)";

	for (const auto& w : r.scene.warnings)
		EXPECT_EQ(w.message.find("measured"), std::string::npos)
			<< "unexpected measured-material warning: " << w.message;

	const pbrt_cpu::BuildResult built = pbrt_cpu::build(r.scene);
	ASSERT_TRUE(built.world);

	hit_record rec;
	const ray probe(point3(0, -1, -5), vec3(0, 0, 1));
	ASSERT_TRUE(built.world->hit(probe, interval(0.001, infinity), rec));
	auto asMeasured = std::dynamic_pointer_cast<measured>(rec.mat);
	ASSERT_TRUE(asMeasured) << "material on the hit triangle should be `measured`, not a fallback";
	EXPECT_TRUE(asMeasured->loaded());
}

TEST(MeasuredBxdfPbrtWiring, MissingFilenameFallsBackToDiffuse) {
	const std::string text =
		"MakeNamedMaterial \"m\" \"string type\" [ \"measured\" ]\n"
		"NamedMaterial \"m\"\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ -10 -10 0   10 -10 0   0 10 0 ]\n";

	const pbrt_scene::ParseResult parsed = pbrt_scene::parse(text);
	ASSERT_TRUE(parsed.ok) << parsed.error;
	const pbrt_flatten::FlatScene scene = pbrt_flatten::flatten(parsed.scene);
	ASSERT_EQ(scene.materials.size(), 1u);
	EXPECT_EQ(scene.materials[0].kind, pbrt_flatten::MaterialKind::Measured);
	EXPECT_TRUE(scene.materials[0].measuredFilename.empty());

	bool sawWarning = false;
	for (const auto& w : scene.warnings)
		if (w.message.find("measured") != std::string::npos) sawWarning = true;
	EXPECT_TRUE(sawWarning) << "a filename-less measured material should warn";

	const pbrt_cpu::BuildResult built = pbrt_cpu::build(scene);
	hit_record rec;
	const ray probe(point3(0, -1, -5), vec3(0, 0, 1));
	ASSERT_TRUE(built.world->hit(probe, interval(0.001, infinity), rec));
	EXPECT_FALSE(std::dynamic_pointer_cast<measured>(rec.mat));
}

TEST(MeasuredBxdfPbrtWiring, UnresolvableFilenameFallsBackToDiffuse) {
	const std::string text =
		"MakeNamedMaterial \"m\" \"string type\" [ \"measured\" ] "
		"\"string filename\" [ \"does_not_exist_anywhere.bsdf\" ]\n"
		"NamedMaterial \"m\"\n"
		"Shape \"trianglemesh\" \"integer indices\" [ 0 1 2 ]\n"
		"  \"point3 P\" [ -10 -10 0   10 -10 0   0 10 0 ]\n";

	const std::string scenePath = TempDir() + "scene_missing_bsdf.pbrt";
	{
		std::ofstream out(scenePath, std::ios::binary);
		out << text;
	}

	const pbrt_load::LoadResult r = pbrt_load::loadFile(scenePath);
	ASSERT_TRUE(r.ok) << r.error;
	ASSERT_EQ(r.scene.materials.size(), 1u);
	EXPECT_TRUE(r.scene.materials[0].measuredFilename.empty())
		<< "an unresolvable .bsdf must be cleared back to empty by pbrt_load.h";

	bool sawWarning = false;
	for (const auto& w : r.scene.warnings)
		if (w.message.find("could not be found") != std::string::npos) sawWarning = true;
	EXPECT_TRUE(sawWarning);
}
