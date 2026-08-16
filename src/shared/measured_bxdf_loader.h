#pragma once
// ---------------------------------------------------------------------------
// measured_bxdf_loader.h -- loads a pbrt-v4 "measured" material's binary
// .bsdf file (Dupuy & Jakob tensor format) into a MeasuredBRDFData.
//
// measured_bxdf.h (this directory) already ports pbrt-v4's PiecewiseLinear2D
// warp sampler and MeasuredBxDF::f/Sample_f/PDF faithfully - what was
// missing was actually getting real data INTO a MeasuredBRDFData from a
// gonioreflectometer-measured .bsdf file on disk, rather than the synthetic
// tables scenes_advanced.h's "Scene 34" placeholder builds by hand.
//
// This is CPU-only (std::ifstream, exceptions-free error propagation via a
// bool return + out-parameter, matching pbrt_load.h's own LoadResult
// convention) and is never included from gpu/optix - measured_bxdf.h and
// materials.h stay untouched and keep compiling exactly as they did before.
//
// Tensor file format (pbrt-v4 src/pbrt/bxdfs.cpp lines 580-829, class Tensor):
//   12-byte magic "tensor_file", 2-byte version {1,0}, 4-byte field count,
//   then per field: 2-byte name length + name, 2-byte ndim, 1-byte dtype,
//   8-byte absolute file offset, ndim x 8-byte shape values. Each field's
//   raw data lives at its own absolute `offset` elsewhere in the file (read
//   via seek, not sequentially) - plain uncompressed little-endian binary,
//   no zlib. This machine is little-endian (x86/x64), so no byte-swapping.
//
// MeasuredBxDFData::Create (pbrt-v4 bxdfs.cpp lines 861-988): reads the 9
// named fields (theta_i, phi_i, ndf, sigma, vndf, spectra, luminance,
// wavelengths, description, jacobian), validates their shapes/dtypes
// against each other, and constructs 4 PiecewiseLinear2D<N> tables plus a
// plain wavelengths array. Ported below in LoadMeasuredBRDF().
//
// One deliberate deviation from MeasuredBRDFData::Build() (measured_bxdf.h):
// that helper passes normalize=true for the ndf/sigma tables. pbrt-v4's own
// construction (bxdfs.cpp ~971-981) passes normalize=false, build_cdf=false
// for ndf and sigma - they are evaluated via Eval() only (never Sample()'d),
// and want their RAW measured values back, not rescaled so the table
// integrates to 1 over the unit square. With normalize=true, piecewise_
// linear_2d.h's build_cdf=false branch replaces the intended "divide by a
// fixed patch-count scale that Eval() exactly cancels back out" with "divide
// by the table's own integral", silently rescaling every ndf/sigma lookup -
// exactly the kind of subtly-wrong-not-crashing bug this feature has no
// forgiving failure mode for. LoadMeasuredBRDF() below does not call
// Build(); it constructs the 5 PiecewiseLinear2D fields directly with
// pbrt-v4's actual flags (false,false / true,true / true,true / false,false
// for ndf / vndf / luminance / spectra respectively - see the Create()
// comment above), so this file's own correctness does not depend on
// Build() at all. Build() itself is left exactly as-is: fixing it is a
// separate, separately-verifiable change against scenes_advanced.h's
// existing "Scene 34" demo and its tests, not part of this loader.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "measured_bxdf.h"   // MeasuredBRDFData, PiecewiseLinear2D<N>

namespace measured_bxdf_io {

// Tensor::Type (pbrt-v4 bxdfs.cpp Tensor::Type enum, line ~583).
enum class TensorType : uint8_t {
	Invalid = 0,
	UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64,
	Float16, Float32, Float64,
};

inline size_t TensorTypeSize(TensorType t) {
	switch (t) {
	case TensorType::UInt8: case TensorType::Int8:    return 1;
	case TensorType::UInt16: case TensorType::Int16:
	case TensorType::Float16:                          return 2;
	case TensorType::UInt32: case TensorType::Int32:
	case TensorType::Float32:                          return 4;
	case TensorType::UInt64: case TensorType::Int64:
	case TensorType::Float64:                          return 8;
	default:                                           return 0;
	}
}

struct TensorField {
	TensorType dtype = TensorType::Invalid;
	std::vector<size_t> shape;
	std::vector<uint8_t> data;   // raw bytes, size = product(shape) * TensorTypeSize(dtype)

	// Reinterprets the raw bytes as a float array. Only meaningful when
	// dtype == Float32 - every numeric field this loader reads (theta_i,
	// phi_i, wavelengths, ndf, sigma, vndf, luminance, spectra) is Float32
	// in every .bsdf file this format's own writer produces (pbrt-v4 never
	// emits Float64 measured data), so this loader does not need a general
	// dtype-conversion path the way a fully generic tensor reader would.
	const float* AsFloat32() const {
		return reinterpret_cast<const float*>(data.data());
	}
};

// Binary parser for the "tensor_file" container format. Mirrors pbrt-v4's
// Tensor class (bxdfs.cpp lines 617-829) field-for-field; the only
// behavioural difference is that a malformed/truncated file returns false
// with an error string instead of calling ErrorExit() and terminating the
// process - this loader runs as part of scene loading, which already has
// an established "warn and fall back" convention (pbrt_load.h) that a hard
// exit would bypass.
class Tensor {
  public:
	bool Load(const std::string& filename, std::string& error) {
		std::ifstream file(filename, std::ios::binary);
		if (!file) {
			error = "unable to open file";
			return false;
		}

		file.seekg(0, std::ios::end);
		const std::streamoff size = file.tellg();
		if (size < 0 || static_cast<uint64_t>(size) < 12 + 2 + 4) {
			error = "too small to be a tensor file";
			return false;
		}
		file.seekg(0, std::ios::beg);

		char header[12];
		uint8_t version[2];
		uint32_t nFields = 0;
		if (!Read(file, header, 12) || !Read(file, version, 2) ||
			!Read(file, &nFields, sizeof(nFields))) {
			error = "truncated header";
			return false;
		}
		if (std::memcmp(header, "tensor_file", 12) != 0) {
			error = "bad magic (not a tensor_file)";
			return false;
		}
		if (version[0] != 1 || version[1] != 0) {
			error = "unsupported tensor_file version";
			return false;
		}

		for (uint32_t i = 0; i < nFields; ++i) {
			uint16_t nameLength = 0, ndim = 0;
			uint8_t dtype = 0;
			uint64_t offset = 0;
			if (!Read(file, &nameLength, sizeof(nameLength))) {
				error = "truncated field name length"; return false;
			}
			std::string name(nameLength, '\0');
			if (nameLength > 0 && !Read(file, &name[0], nameLength)) {
				error = "truncated field name"; return false;
			}
			if (!Read(file, &ndim, sizeof(ndim)) ||
				!Read(file, &dtype, sizeof(dtype)) ||
				!Read(file, &offset, sizeof(offset))) {
				error = "truncated field header"; return false;
			}
			if (dtype == 0 || dtype > static_cast<uint8_t>(TensorType::Float64)) {
				error = "field '" + name + "' has an unknown dtype"; return false;
			}

			std::vector<size_t> shape(ndim);
			size_t totalSize = TensorTypeSize(static_cast<TensorType>(dtype));
			for (uint16_t j = 0; j < ndim; ++j) {
				uint64_t dim = 0;
				if (!Read(file, &dim, sizeof(dim))) {
					error = "truncated field shape"; return false;
				}
				shape[j] = static_cast<size_t>(dim);
				totalSize *= shape[j];
			}

			// The field's data lives at its own absolute offset, not here -
			// seek there, read it, then seek back to keep walking the
			// header table for the next field.
			const std::streamoff resumeAt = file.tellg();
			file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
			std::vector<uint8_t> data(totalSize);
			if (totalSize > 0 && !Read(file, data.data(), totalSize)) {
				error = "field '" + name + "' data could not be read (offset " +
						std::to_string(offset) + ", " + std::to_string(totalSize) + " bytes)";
				return false;
			}
			file.seekg(resumeAt, std::ios::beg);

			TensorField field;
			field.dtype = static_cast<TensorType>(dtype);
			field.shape = std::move(shape);
			field.data = std::move(data);
			fields_.emplace(std::move(name), std::move(field));
		}
		return true;
	}

	const TensorField* Field(const std::string& name) const {
		auto it = fields_.find(name);
		return (it == fields_.end()) ? nullptr : &it->second;
	}

  private:
	static bool Read(std::ifstream& f, void* dst, size_t bytes) {
		if (bytes == 0) return true;
		f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
		return static_cast<bool>(f);
	}

	std::map<std::string, TensorField> fields_;
};

namespace detail {
inline constexpr double kPi = 3.14159265358979323846;
}

// Reads a .bsdf file and fills `out` with its 4 warp tables + wavelength
// array, exactly as pbrt-v4's MeasuredBxDFData::Create does (bxdfs.cpp
// lines 861-988) - field lookup, shape/dtype cross-validation, isotropic
// detection, then direct construction of the PiecewiseLinear2D tables
// (bypassing MeasuredBRDFData::Build() - see this file's header comment for
// why). Returns false with `error` set on any structural problem; `out` is
// left default-constructed (unusable, matching a null MeasuredBRDFData*) in
// that case.
inline bool LoadMeasuredBRDF(const std::string& filename, MeasuredBRDFData& out,
							 std::string& error) {
	Tensor tf;
	if (!tf.Load(filename, error)) return false;

	const char* kRequired[] = {"theta_i", "phi_i", "ndf", "sigma", "vndf",
								"spectra", "luminance", "wavelengths",
								"description", "jacobian"};
	const TensorField* f[10];
	for (int i = 0; i < 10; ++i) {
		f[i] = tf.Field(kRequired[i]);
		if (!f[i]) {
			error = std::string("missing field '") + kRequired[i] + "'";
			return false;
		}
	}
	const TensorField &theta_i = *f[0], &phi_i = *f[1], &ndf = *f[2], &sigma = *f[3],
					  &vndf = *f[4], &spectra = *f[5], &luminance = *f[6],
					  &wavelengths = *f[7], &description = *f[8], &jacobian = *f[9];

	const auto isF32 = [](const TensorField& x) { return x.dtype == TensorType::Float32; };
	const auto isU8  = [](const TensorField& x) { return x.dtype == TensorType::UInt8; };

	// Mirrors pbrt-v4's single big validation expression (bxdfs.cpp
	// ~902-925) field for field.
	if (!(description.shape.size() == 1 && isU8(description) &&

		  theta_i.shape.size() == 1 && isF32(theta_i) &&

		  phi_i.shape.size() == 1 && isF32(phi_i) &&

		  wavelengths.shape.size() == 1 && isF32(wavelengths) &&

		  ndf.shape.size() == 2 && isF32(ndf) &&

		  sigma.shape.size() == 2 && isF32(sigma) &&

		  vndf.shape.size() == 4 && isF32(vndf) &&
		  vndf.shape[0] == phi_i.shape[0] && vndf.shape[1] == theta_i.shape[0] &&

		  luminance.shape.size() == 4 && isF32(luminance) &&
		  luminance.shape[0] == phi_i.shape[0] &&
		  luminance.shape[1] == theta_i.shape[0] &&
		  luminance.shape[2] == luminance.shape[3] &&

		  isF32(spectra) && spectra.shape.size() == 5 &&
		  spectra.shape[0] == phi_i.shape[0] && spectra.shape[1] == theta_i.shape[0] &&
		  spectra.shape[2] == wavelengths.shape[0] &&
		  spectra.shape[3] == spectra.shape[4] &&

		  luminance.shape[2] == spectra.shape[3] &&
		  luminance.shape[3] == spectra.shape[4] &&

		  jacobian.shape.size() == 1 && jacobian.shape[0] == 1 && isU8(jacobian))) {
		error = "invalid .bsdf file structure (fields present but shapes/dtypes "
				"don't match pbrt-v4's MeasuredBxDFData layout)";
		return false;
	}

	const bool isotropic = phi_i.shape[0] <= 2;
	const float* phiPtr = phi_i.AsFloat32();
	const float* thetaPtr = theta_i.AsFloat32();
	const float* wlPtr = wavelengths.AsFloat32();

	if (!isotropic) {
		// pbrt-v4: reduction = round((2*Pi) / (phi_i[last] - phi_i[0])); only
		// reduction == 1 (a full 0..2*Pi sweep, no symmetry folding) is
		// supported - no scene in this loader's corpus uses a reduced
		// anisotropic BRDF, so this is reported rather than silently
		// mishandled.
		const double span = static_cast<double>(phiPtr[phi_i.shape[0] - 1]) - phiPtr[0];
		const int reduction = span != 0.0
			? static_cast<int>(std::lround((2.0 * detail::kPi) / span))
			: 0;
		if (reduction != 1) {
			error = "anisotropic .bsdf with an unsupported phi_i reduction (" +
					std::to_string(reduction) + " != 1)";
			return false;
		}
	}

	out.isotropic = isotropic;

	// NDF / sigma: unconditional 2D tables, Eval()-only (never sampled), so
	// normalize=false, build_cdf=false - see this file's header comment.
	out.ndf = PiecewiseLinear2D<0>(ndf.AsFloat32(),
									static_cast<int>(ndf.shape[1]),
									static_cast<int>(ndf.shape[0]),
									/*normalize=*/false, /*build_cdf=*/false);
	out.sigma = PiecewiseLinear2D<0>(sigma.AsFloat32(),
									  static_cast<int>(sigma.shape[1]),
									  static_cast<int>(sigma.shape[0]),
									  /*normalize=*/false, /*build_cdf=*/false);

	const int paramRes[2] = {static_cast<int>(phi_i.shape[0]), static_cast<int>(theta_i.shape[0])};
	const float* paramVals[2] = {phiPtr, thetaPtr};

	// VNDF / luminance: conditioned on (phi_i, theta_i), genuinely sampled
	// (Sample()/Invert()), so normalize=true, build_cdf=true (the defaults).
	out.vndf = PiecewiseLinear2D<2>(vndf.AsFloat32(),
									 static_cast<int>(vndf.shape[3]),
									 static_cast<int>(vndf.shape[2]),
									 paramRes, paramVals);
	out.luminance = PiecewiseLinear2D<2>(luminance.AsFloat32(),
										  static_cast<int>(luminance.shape[3]),
										  static_cast<int>(luminance.shape[2]),
										  paramRes, paramVals);

	// Spectra: conditioned on (phi_i, theta_i, wavelength), Eval()-only, so
	// normalize=false, build_cdf=false, same reasoning as ndf/sigma.
	const int specRes[3] = {static_cast<int>(phi_i.shape[0]), static_cast<int>(theta_i.shape[0]),
							 static_cast<int>(wavelengths.shape[0])};
	const float* specVals[3] = {phiPtr, thetaPtr, wlPtr};
	out.spectra = PiecewiseLinear2D<3>(spectra.AsFloat32(),
										static_cast<int>(spectra.shape[4]),
										static_cast<int>(spectra.shape[3]),
										specRes, specVals,
										/*normalize=*/false, /*build_cdf=*/false);

	out.wavelengths.assign(wlPtr, wlPtr + wavelengths.shape[0]);
	return true;
}

// Loads (or returns a previously-loaded) MeasuredBRDFData for `resolvedPath`,
// keyed by that exact path string. Mirrors pbrt-v4's own
// MeasuredBxDF::BRDFDataFromFile caching (bxdfs.cpp, `static std::map<...>
// loadedData`) - the sportscar scene this loader was built against
// references ilm_l3_37_metallic_spec.bsdf (7MB) from 4 different materials,
// and without this every one of them would parse the same file again.
//
// A null return means loading failed; `error` carries why. A failure is
// cached too (as a null entry) so a bad or missing file is only ever
// attempted once, not once per material that names it - the alternative,
// re-opening the same 7-15MB path over and over just to fail the same way,
// is not "graceful" in any sense that matters.
//
// The function-local statics are shared across every translation unit that
// includes this header (guaranteed for an inline function's local statics,
// unlike a namespace-scope `static` variable, which would give each TU its
// own copy - an ODR-safe but cache-defeating outcome). The mutex covers the
// case of scene loading and/or test execution happening on more than one
// thread at once; ordinary single-threaded scene loading pays an
// uncontended lock for that safety.
inline std::shared_ptr<const MeasuredBRDFData> GetMeasuredBRDFDataCached(
		const std::string& resolvedPath, std::string& error) {
	static std::mutex mtx;
	static std::map<std::string, std::shared_ptr<const MeasuredBRDFData>> cache;

	std::lock_guard<std::mutex> lock(mtx);
	auto it = cache.find(resolvedPath);
	if (it != cache.end()) {
		if (!it->second) error = "previously failed to load (see earlier warning)";
		return it->second;
	}

	auto data = std::make_shared<MeasuredBRDFData>();
	std::string err;
	if (!LoadMeasuredBRDF(resolvedPath, *data, err)) {
		error = err;
		cache.emplace(resolvedPath, nullptr);
		return nullptr;
	}
	std::shared_ptr<const MeasuredBRDFData> result = data;
	cache.emplace(resolvedPath, result);
	return result;
}

} // namespace measured_bxdf_io
