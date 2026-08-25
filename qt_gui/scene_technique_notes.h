#pragma once
#include <QCoreApplication>
#include <QDebug>
#include <QHash>
#include <QSet>
#include <QString>
#include <QStringList>

// scene_technique_notes.h -- per-scene "what rendering technique does this
// demonstrate" text for the Basic Settings tab's dynamic info icon (see
// MainWindow::updateSceneTechInfoIcon(), mainwindow_style.cpp).
//
// Unlike the ~27 other info-icon tooltips in this app (fixed text, written
// once at construction), this content is looked up fresh per selected
// scene. It lives here rather than as a new SceneDescriptor field/
// scene_metadata.dll export because it's pure GUI documentation, not
// renderer behavior - the same reasoning every other info-icon tooltip is
// GUI-only text, never round-tripped through the DLL.
//
// Scoped to self-contained scenes only (requires_files == false): the 45
// compiled-in procedural scenes (src/TheRestOfYourLife/scenes*.h) plus the
// 31 curated pbrt-example scenes (bundled pbrt_scenes/*.pbrt files,
// registered via pbrt_scene_registry::build_curated_pbrt_scene_descriptor()
// in scene_registry.h, which hardcodes requires_files=false). A scene id
// with no entry here - every "Requires External Files" scene - falls back
// to an honest "not written yet" message in forScene() below, rather than
// silently showing nothing or stale content.
namespace scene_technique_notes {

inline const QHash<QString, QString>& notes() {
	static const QHash<QString, QString> kNotes = {
		// ---------------------------------------------------------------
		// Basics (A1-A9, minus A4)
		// ---------------------------------------------------------------
		{"A1", QObject::tr(R"note(Builds the box from shared Cornell-box data: five lambertian walls, a diffuse_light ceiling quad, a rotated axis-aligned box in plain white lambertian, and a sphere with a dielectric BSDF doing full Fresnel-weighted reflection/refraction. It's the renderer's canonical convergence test - diffuse interreflection off the walls, a specular/transmissive object for BSDF-sampling correctness, and an area light quad that exercises next-event estimation, so most cross-cutting material or light-sampling regressions show up here first.)note")},
		{"A2", QObject::tr(R"note(The classic "Ray Tracing in One Weekend" closer: a checker-textured lambertian ground sphere under a BVH-accelerated field of small spheres, most diffuse but some built with a second center offset in time so the sphere itself carries linear motion blur, some metal with per-sphere random fuzz radius, and a few dielectric. Three large feature spheres - a sharp dielectric, a flat lambertian, and a zero-fuzz mirror-perfect metal - sit in front as reference specular/diffuse endpoints. Primarily a stress test for the BVH build and for motion-blurred ray-sphere intersection alongside the fuzzy-metal reflection model.)note")},
		{"A3", QObject::tr(R"note(Two enormous spheres (radius 10, one above and one below the origin) share a single procedural checker texture sampled through spherical UV coordinates and shaded with plain lambertian - the pattern is generated analytically from 3D position, not an image lookup. A handful of smaller accent spheres (diffuse, glossy metal with slight fuzz, and a dielectric) sit on the lower sphere's cap for scale and to give the frame real specular/refractive behavior. Mostly a test of the procedural texture's 3D-to-checker mapping holding up at large sphere-surface curvature.)note")},
		{"A5", QObject::tr(R"note(Ground and a large sphere both use a Perlin/turbulence-based procedural marble texture evaluated directly from world-space position and fed through lambertian - no image texture involved. Two smaller companion spheres reuse the same technique at a higher noise frequency for contrast, lit by a diffuse_light quad sampled via next-event estimation. Tests the Perlin-noise turbulence/marble output feeding correctly into the BSDF pipeline under real NEE lighting rather than flat ambient.)note")},
		{"A6", QObject::tr(R"note(Five quad primitives, each spanning a different orientation (front-facing, side-facing, top-facing) via distinct basis vectors, each with a flat lambertian color - this exercises the quad hittable's plane-intersection and per-orientation normal/UV computation rather than shading complexity. A separate diffuse_light quad floats in the room as the scene's only emitter, giving next-event estimation something concrete to importance-sample. Mainly a correctness check for quad geometry and orientation-dependent normals under real area-light sampling.)note")},
		{"A7", QObject::tr(R"note(Reuses the Perlin-noise ground/sphere setup from the marble-texture scenes, but the real subject is emission: a sphere given a diffuse_light material (warm tint) acts as a curved, non-planar area-light emitter, alongside a separate cool-toned diffuse_light quad. Having both a spherical and a quad emitter live in the same scene tests that light-sampling handles solid-angle sampling correctly across different emitter geometries, not just the usual flat quad case - the color-temperature split makes it easy to see which light contributes where.)note")},
		{"A8", QObject::tr(R"note(Rebuilds the standard Cornell walls plus its own oversized ceiling diffuse_light quad, then replaces the usual solid boxes with two rotated/translated boxes wrapped in constant_medium - a homogeneous isotropic-scattering participating medium with low density and its own tint, instead of the classic monochrome smoke. This is the volumetric-medium test: box geometry used elsewhere as solid lambertian surfaces here instead defines the boundary of an isotropic in-scattering volume, checking that medium sampling and boundary-shape intersection cooperate correctly.)note")},
		{"A9", QObject::tr(R"note(The most feature-dense scene in the set: a grid of randomly-heighted lambertian boxes forming undulating BVH-built ground, a ceiling-quad area light, a motion-blurred lambertian sphere, a dielectric glass sphere, a fuzzy metal sphere, a colored constant_medium bounded inside a dielectric sphere (a self-contained volumetric fog pocket), a second near-zero-density constant_medium wrapped around a giant enclosing sphere as faint global atmosphere, a mipmapped image-textured earth sphere, a Perlin-noise marble sphere, and a cluster of small white spheres packed into a rotated BVH sub-tree. Exercises motion blur, dielectric/metal/lambertian BSDFs, bounded and global participating media, image-texture mip-mapping, and multi-level BVH construction all in one render.)note")},

		// ---------------------------------------------------------------
		// Materials (B1-B14, B23) - procedural
		// ---------------------------------------------------------------
		{"B1", QObject::tr(R"note(A row of five spheres running through pure roughness values on the same gold-tinted rough_metal material (GGX/Trowbridge-Reitz microfacet distribution with a simple RGB-tinted reflectance, not full complex-IOR Fresnel). Isolated on a ground plane under one area light, so the only variable in frame is how the specular highlight spreads and dims as roughness increases.)note")},
		{"B2", QObject::tr(R"note(The same rough_metal GGX BxDF as the rough-metal-spheres scene (brushed aluminium box, brushed gold sphere at a different roughness) but dropped into the familiar Cornell box instead of an open studio setup, so glossy microfacet reflection has to hold up under real indirect bounce lighting and colored wall bleed rather than a single overhead light.)note")},
		{"B3", QObject::tr(R"note(A Cornell box glass sphere swapped from perfect dielectric to rough_dielectric - GGX microfacets applied to both the reflection and the transmission lobe, so refracted light scatters into a frosted blur instead of a sharp Snell-law ray, while still respecting Fresnel's reflect/transmit split per microfacet.)note")},
		{"B4", QObject::tr(R"note(Polished gold and aluminium surfaces using the conductor material with real per-channel complex Fresnel from measured eta/k spectra, rather than an RGB-albedo approximation - so the metal color comes from actual reflectance spectra, not an artist-picked tint.)note")},
		{"B5", QObject::tr(R"note(A coated-diffuse sphere and box using coated_diffuse: a rough dielectric coat layered over a Lambertian base. Exercises the layered-BSDF solve - light can specularly reflect off the coat, or transmit through and pick up the diffuse base color, with internal reflection between the two layers.)note")},
		{"B6", QObject::tr(R"note(A vertical glass panel using thin_dielectric splitting the Cornell box in two: zero-thickness glass with the analytic multi-bounce Fresnel formula instead of a real refractive interface, so transmitted rays pass straight through with no bending. The panel is tilted deliberately, since IOR-1.5 Fresnel reflectance only becomes visible near grazing incidence.)note")},
		{"B7", QObject::tr(R"note(Lacquered-metal look via coated_conductor: a gold sphere and copper box, each a rough dielectric coat over a GGX conductor base with per-channel complex Fresnel. The most stacked BxDF in this category - it combines spectral metal reflectance underneath an achromatic glossy coat, showing the coat's Fresnel sheen sitting on top of the metal's own tint.)note")},
		{"B8", QObject::tr(R"note(A wax-like sphere using diffuse_transmission with separate reflectance and transmittance colors - light scatters diffusely into both the same hemisphere (reflection) and the opposite one (transmission), a cheap two-parameter stand-in for translucency that needs no volumetric random walk, unlike the subsurface-slab scene's approach to the same visual goal.)note")},
		{"B9", QObject::tr(R"note(A crystal sphere using normalized_fresnel - Fresnel-weighted diffuse exitance where the BSDF value rises toward grazing angles (where Fresnel transmittance is low, so more internally-scattered light escapes). Sampling stays cosine-hemisphere, but the scattering PDF carries the Fresnel weighting itself, normalized so total reflected energy stays correct under MIS.)note")},
		{"B10", QObject::tr(R"note(Seven spheres sweeping the principled material's parameter space in one row: matte diffuse, low-roughness plastic, clearcoated plastic, semi-metallic, rough near-metallic, smooth full metal, and a clearcoated full metal - varying metallic, roughness, and clearcoat independently to show how one unified BSDF spans the whole dielectric-to-conductor range that separate materials elsewhere in this category cover individually.)note")},
		{"B11", QObject::tr(R"note(Five spheres using the hair material (HairBxDF) with distinct absorption/roughness parameters, tuned for a strongly forward/specular-scattering fiber BSDF rather than a diffuse one - the light had to be recalibrated far dimmer than other scenes in this set because hair's peak BSDF response blows out under normal area-light intensity. Applied to sphere surfaces rather than real curve geometry, isolating the BxDF's shading behavior from the curve-intersection machinery (see the Curve Fibers scene for that half).)note")},
		{"B12", QObject::tr(R"note(Two perturbation techniques on plain Lambertian surfaces: bump mapping displaces shading normals from a Perlin marble noise texture (back wall and box), while normal mapping perturbs them from a checker pattern (sphere) - the geometric surface stays flat/spherical but the shading normal used for lighting is bent to fake fine surface detail.)note")},
		{"B13", QObject::tr(R"note(Not a true BSSRDF - light entering the milky wax slab and jade sphere passes through a dielectric shell into a constant_medium (homogeneous participating medium with its own density and tint), so the translucent glow comes from a real volumetric random walk inside the object bounded by refractive Fresnel at the surface, rather than a diffusion-based subsurface term.)note")},
		{"B14", QObject::tr(R"note(Exercises the pbrt-v4 measured-BRDF data pipeline - synthetically generated tabulated NDF, sigma, VNDF, luminance and per-wavelength spectral tables (rather than a real captured material) across a row of five spheres. BSDF evaluation itself is simplified to cosine-hemisphere sampling with a flat tint rather than the full 5D importance-sampling warp chain, so this scene tests the tabulated-data plumbing more than a realistic measured-material result.)note")},
		{"B23", QObject::tr(R"note(A literal triangular glass prism using a dispersive dielectric - a Cauchy-equation index of refraction that varies with wavelength, matching real crown glass - instead of a flat IOR. White light entering the prism splits by wavelength on exit, fanning across a catcher screen the way a physical glass prism does, but only under --spectral, which tracks per-wavelength rays through the renderer; without it, each RGB channel just refracts by a fixed, slightly different amount rather than a continuous spread.)note")},

		// ---------------------------------------------------------------
		// Materials (B15-B22) - curated pbrt examples
		// ---------------------------------------------------------------
		{"B15", QObject::tr(R"note(pbrt's "mix" material blends two materials per shading point using real per-sample stochastic selection - a probability-weighted choice at each hit, not a flat blended color - so a mix of matte diffuse and a metallic conductor shows up as genuine speckled grain: some samples land pure diffuse, others pure specular, averaging out correctly over many samples. Exercised identically on CPU, GPU-recursive, and GPU-wavefront.)note")},
		{"B16", QObject::tr(R"note(Four material kinds bundled nowhere else in this app: thindielectric (a zero-thickness glass sheet that refracts without displacing the ray, unlike ordinary dielectric), coatedconductor (a clear coat over a metal base), diffusetransmission (light passes through as well as scattering back, like a thin leaf or paper), and subsurface scattering via a named preset ("Marble") - real subsurface parameters without needing an external measured-data file.)note")},
		{"B17", QObject::tr(R"note(pbrt's CoatedDiffuse material (a diffuse base under a clear dielectric coat, like varnished wood) reading its reflectance from an actual image texture instead of a flat color - this exact combination used to silently ignore the texture and fall back to solid grey on both backends.)note")},
		{"B18", QObject::tr(R"note(A conductor material driven by explicit RGB eta/k (the complex index of refraction that gives metals their tinted, wavelength-dependent reflectance) rather than a named preset - plus the same complex-IOR math applied through a coatedconductor's metal base layer, producing accurate colored specular highlights instead of a flat mirror tint.)note")},
		{"B19", QObject::tr(R"note(DiffuseTransmission (light both reflects and transmits diffusely, the model for something like a thin leaf or lampshade) with its reflectance and transmittance each bound to a real texture instead of flat colors, on both backends.)note")},
		{"B20", QObject::tr(R"note(pbrt's Material "hair" - the Marschner/Chiang fiber-scattering model, the same physically-based hair BSDF real film production renderers use - applied to plain spheres here for a controlled, side-by-side comparison against this project's own native Hair Fibers demo.)note")},
		{"B21", QObject::tr(R"note(A pbrt checkerboard texture whose two colors are themselves each bound to a real image (an imagemap texture nested one level inside the checker/mix texture), not just flat literal colors - tests that texture references can compose, not just appear standalone.)note")},
		{"B22", QObject::tr(R"note(pbrt's NamedMaterial referenced directly on a shape (declared once, reused by name, rather than only appearing as a "mix" material's sub-ingredient) plus a texture bound to one of that material's parameters, and an AreaLightSource with its twosided flag set so both faces emit.)note")},

		// ---------------------------------------------------------------
		// Lights (C1-C7) - procedural
		// ---------------------------------------------------------------
		{"C1", QObject::tr(R"note(An open scene lit purely by a procedural HDR sky gradient (blue zenith fading to a warm horizon) wired in as an image-based infinite light, rather than any local emitter. A diffuse sphere, a near-mirror metal sphere, and a dielectric sphere sit under it to show how that environment illumination reads across different BSDFs, from soft ambient-occlusion-like shading to sharp reflected/refracted copies of the sky gradient.)note")},
		{"C2", QObject::tr(R"note(Bare Cornell walls lit entirely by a single spotlight-style punctual light aimed straight down from the ceiling, with a wide total cone width and a narrower inner falloff start, giving it a smooth penumbra rather than a hard-edged disc of light. Unlike the area-light Cornell scenes, this is a delta light with no surface to sample - the cone shape and falloff are the whole point.)note")},
		{"C3", QObject::tr(R"note(Bare Cornell walls lit by a parallel, sun-like distant light: a fixed direction vector with no 1/r-squared falloff, so the radiance scale directly is the incident irradiance rather than an intensity compensating for distance. A deliberate contrast case for verifying the punctual-light pipeline handles directional, non-attenuating lights correctly alongside the point/spot/gonio lights that do fall off.)note")},
		{"C4", QObject::tr(R"note(Cornell walls lit by a single overhead isotropic point light with classic 1/r-squared intensity falloff - the simplest punctual emitter in this category, with no cone, no directional profile, no parallel rays, just distance-based attenuation from one location near the ceiling.)note")},
		{"C5", QObject::tr(R"note(Cornell walls lit by a goniometric light: a point light whose intensity varies by direction according to a tabulated image rather than emitting uniformly, combined with the same 1/r-squared falloff as an ordinary point light and an explicit rotation orienting the profile in world space. The arbitrary-directional-intensity case, as opposed to a spotlight's simple cone or a point light's uniform emission.)note")},
		{"C6", QObject::tr(R"note(Cornell walls lit by a projection light: a punctual light that projects a checkerboard slide image through a perspective frustum, casting a sharp patterned beam onto the scene like a real slide projector rather than emitting a plain cone or uniform sphere of light.)note")},
		{"C7", QObject::tr(R"note(A Cornell-style room with no ceiling light quad at all - instead the back wall has an actual rectangular hole cut into it, and a sky light is visible only through that window. Exercises portal-sampled infinite lighting: the environment light's sampling is restricted to directions actually visible through the portal aperture rather than the full sphere, the harder, more failure-prone case for infinite-light next-event estimation.)note")},

		// ---------------------------------------------------------------
		// Lights (C8-C15) - curated pbrt examples
		// ---------------------------------------------------------------
		{"C8", QObject::tr(R"note(All five of pbrt-v4's punctual (delta-distribution) light kinds in one scene: point, spot, distant, goniometric, and projection - a single reference scene for comparing every zero-area light type side by side.)note")},
		{"C9", QObject::tr(R"note(Real image decoding for pbrt's goniometric and projection lights, which previously silently ignored their own filename parameter and fell back to a uniform beam - this scene is the regression check that they now actually read and apply the image.)note")},
		{"C10", QObject::tr(R"note(pbrt's "blackbody L" light color, specified as a temperature in Kelvin rather than an RGB triple, converted through real Planckian-locus blackbody-to-RGB math. Two otherwise-identical panels at a warm, incandescent-like temperature and a cool, overcast-sky-like one sit side by side so a color regression would be immediately visible as both panels going flat white.)note")},
		{"C11", QObject::tr(R"note(An AreaLightSource on a non-triangle shape (sphere or quad) that's both filename-textured (its emission pattern comes from an image, not a flat color) and twosided (emits from both faces, not just the one its surface normal points toward) - on both backends.)note")},
		{"C12", QObject::tr(R"note(pbrt's "infinite" light - a constant-color light with no shape at all, illuminating the scene from every direction at once, the simplest possible substitute for a sky. Needs open geometry to see the effect, since a closed room would block it from every side anyway.)note")},
		{"C13", QObject::tr(R"note(Disk and cylinder shapes acting as real next-event-estimation light sources - solid-angle sampling toward the light on every bounce, not just accidentally hitting it - on both GPU backends, converging as cleanly as the CPU path instead of the noisier "hope a ray hits it" fallback non-NEE lights get.)note")},
		{"C14", QObject::tr(R"note(Two separate sphere area lights in one scene - a minimal case that once pinned a real GPU bug where a light-type lookup table was sized for only one light, so every light past the first silently misread its own type and rendered wrong.)note")},
		{"C15", QObject::tr(R"note(An area light shaped as an irregular 5-triangle fan - not a simple parallelogram, and not something the renderer's quad-merge optimization (which re-joins two triangles into one quad for cheaper sampling) can rejoin - so this exercises genuine per-triangle light sampling on the GPU instead of the more common quad shortcut.)note")},

		// ---------------------------------------------------------------
		// Cameras (D1-D8) - procedural
		// ---------------------------------------------------------------
		{"D1", QObject::tr(R"note(An open scene (checker ground plus spheres at several depths, one glass, one metal) rendered with the default perspective camera's built-in thin-lens defocus blur, so only the sphere at the focus distance renders sharp while the near and far spheres blur out - the straightforward aperture-plus-focus-distance depth-of-field path, no alternate camera class involved.)note")},
		{"D2", QObject::tr(R"note(A row of spheres over a checker ground, rendered through an explicit orthographic camera. Rays are cast parallel rather than converging from a single eye point, so the spheres show no perspective foreshortening - equal-size spheres stay equal-size regardless of depth, the defining visual signature of parallel projection.)note")},
		{"D3", QObject::tr(R"note(A ring of colored spheres around a central emissive sphere, captured by an explicit spherical (equirectangular) camera, so the whole 360-degree surroundings are mapped into one panoramic image rather than a bounded field-of-view frustum.)note")},
		{"D4", QObject::tr(R"note(Five spheres at increasing depth over a checker ground, rendered through an explicit realistic camera built from a real multi-element lens prescription (per-surface curvature, thickness, IOR, and aperture) rather than an idealized thin lens - rays are traced element-by-element through the lens stack, producing genuine lens-induced bokeh and vignetting instead of a closed-form defocus-disc approximation.)note")},
		{"D5", QObject::tr(R"note(The same classic Cornell box geometry as the plain Cornell Box scene and the other camera-comparison Cornell scenes, rendered with the default perspective camera's thin-lens defocus blur rather than an alternate camera class. Isolates depth-of-field as the one variable against the same room the other camera-comparison scenes also use, so the technique's effect can be compared directly rather than confounded with different geometry.)note")},
		{"D6", QObject::tr(R"note(The same shared Cornell box as the other camera-comparison scenes, rendered through an explicit orthographic camera, viewed dead-on down the box's depth axis. Because projection is parallel rather than perspective, the box's edges stay parallel all the way to the frame edges instead of converging the way the perspective/depth-of-field version does - the same room used specifically to make that contrast legible.)note")},
		{"D7", QObject::tr(R"note(The same shared Cornell box, toured with an explicit spherical (equirectangular) camera positioned at the box's center. The panorama wraps all five walls, the ceiling light, and the glass sphere into one 360-degree image from a single interior vantage point - the same technique as the standalone Spherical Camera scene, now applied to the shared comparison geometry.)note")},
		{"D8", QObject::tr(R"note(The same shared Cornell box, rendered through the identical multi-element realistic-camera lens prescription used in the standalone Realistic Camera scene, but with its aperture scaled up to suit the box's much larger scale - keeping the defocus cone's angular size, and thus the visible lens bokeh, comparable despite the different scene scale.)note")},

		// ---------------------------------------------------------------
		// Cameras (D9-D12) - curated pbrt examples
		// ---------------------------------------------------------------
		{"D9", QObject::tr(R"note(A perspective camera's thin-lens depth-of-field - lens radius and focus distance - loaded straight from a pbrt file's Camera directive, on both backends. The same optical model as this project's own native Depth of Field scene, exercising the file-loading path instead.)note")},
		{"D10", QObject::tr(R"note(pbrt's orthographic (parallel-projection) camera loaded from a file - rays are all parallel instead of converging at an eye point, so two same-size spheres at different depths read as the same size on screen rather than the nearer one looking bigger, the opposite of ordinary perspective foreshortening.)note")},
		{"D11", QObject::tr(R"note(pbrt's spherical (equal-area) camera loaded from a file - rays fan out in every direction from a single point rather than through a flat image plane, so one render can capture a full 360-degree surround. Placed inside an enclosed room so every direction actually has something to see.)note")},
		{"D12", QObject::tr(R"note(pbrt's realistic camera - a real multi-element lens system, not the idealized thin-lens/pinhole model every other camera scene uses - loaded from a lens-prescription file, tracing rays through actual glass elements for authentic depth-of-field and lens aberrations, including the lensfile-loading path a compiled-in scene never exercised.)note")},

		// ---------------------------------------------------------------
		// Volumes (E1-E4) - procedural
		// ---------------------------------------------------------------
		{"E1", QObject::tr(R"note(Fills a Cornell box with a homogeneous participating medium - a boundary box with a uniform scattering coefficient and Henyey-Greenstein phase function. Every point inside the boundary has identical density, so this tests the basic free-flight-sampling / single-scattering-coefficient medium path rather than any spatially-varying density model.)note")},
		{"E2", QObject::tr(R"note(A real heterogeneous volume whose density is evaluated procedurally per-point via multi-octave Perlin noise with a wispiness perturbation and an altitude falloff, rather than a lookup into baked voxel data. Rendering uses delta tracking (null-collision sampling) to handle the non-uniform density field, giving a denser base and thinning top the way a real cloud does.)note")},
		{"E3", QObject::tr(R"note(Shows the dielectric-plus-internal-medium combination as its own subject: glass spheres, each built from a refractive dielectric surface with a homogeneous participating medium of a different density packed inside it. Exercises the case where a scattering medium is bounded by a real refracting surface rather than an opaque shell, so rays must refract in, scatter/absorb through the internal fog, and refract back out.)note")},
		{"E4", QObject::tr(R"note(A heterogeneous medium with real per-voxel scattering data baked into a 3D grid, with independent red/green/blue channels generated at different frequencies so the color decorrelates spatially instead of reading as a uniformly tinted cloud - sampled via majorant-grid-accelerated delta tracking. Unlike a procedural medium evaluated analytically per point, this is discrete voxel data mapped from the render-space box onto the grid's own coordinate space.)note")},

		// ---------------------------------------------------------------
		// Volumes (E5-E8) - curated pbrt examples
		// ---------------------------------------------------------------
		{"E5", QObject::tr(R"note(pbrt's MakeNamedMedium "cloud" - a heterogeneous scattering volume whose density comes from 3D Perlin noise, the same noise-driven approach real cloud rendering uses to avoid a flat, obviously-fake fog block - loaded from a file, on both backends.)note")},
		{"E6", QObject::tr(R"note(A homogeneous participating medium (uniform fog/scattering density) attached to a cylinder shape - this combination used to silently render as ordinary empty geometry with no fog effect at all on the GPU backends; this scene is the regression check that the medium is now real there too.)note")},
		{"E7", QObject::tr(R"note(pbrt's MakeNamedMedium "rgbgrid" - a 3D voxel grid where each cell carries its own RGB scattering color, not just a density scalar - rendering as a soft, coloured nebula-like volume, on both backends.)note")},
		{"E8", QObject::tr(R"note(pbrt's MakeNamedMedium "uniformgrid" - a single-channel density voxel grid, no per-voxel color, unlike the RGB grid medium scene - rendering as a soft glowing blob, the simpler sibling of that scene.)note")},

		// ---------------------------------------------------------------
		// Geometry (F1, F2, F4) - procedural
		// ---------------------------------------------------------------
		{"F1", QObject::tr(R"note(Tests the bilinear patch primitive - a true 4-corner curved surface defined by its own intersection routine, not a pair of triangles - by placing two patches whose corners are offset in height to form a hyperbolic-paraboloid saddle and a curved ramp. A soft-roughness metal material is used deliberately so the shading gradient traces the surface's continuous curvature - a mirror finish would only show a single sharp highlight and hide the fact that the surface isn't flat.)note")},
		{"F2", QObject::tr(R"note(Builds a procedural icosahedron as a real indexed triangle mesh, exercising the mesh's watertight ray-triangle intersection path used by every loaded mesh in the renderer. No per-vertex normals are supplied, so each triangle falls back to its flat per-face geometric normal, giving the faceted low-poly look rather than smooth-shaded interpolated normals.)note")},
		{"F4", QObject::tr(R"note(Exercises real curve geometry with its own ray-curve intersection test, unlike the separate Hair Fibers scene which shades plain spheres with a hair BSDF instead of curving the surface itself. Bezier strands with root-to-tip taper are rooted across the surface in a scattered pattern, so this tests the curve primitive's actual swept-ribbon geometry rather than the hair shading model.)note")},

		// ---------------------------------------------------------------
		// Geometry (F5-F10) - curated pbrt examples
		// ---------------------------------------------------------------
		{"F5", QObject::tr(R"note(pbrt's Shape "plymesh" real per-vertex UV data, threaded through both backends - previously GPU-recursive rendered this exact scene solid black because those UVs were silently dropped.)note")},
		{"F6", QObject::tr(R"note(pbrt's Shape "plymesh" loading a real external .ply mesh file end to end, including fan-triangulating a face in the source file that isn't already a triangle - the loader can't assume every face is pre-triangulated.)note")},
		{"F7", QObject::tr(R"note(pbrt's Shape "curve" - true cubic-Bezier fiber geometry, tessellated into GPU-friendly triangles under the hood, rather than any triangle-mesh approximation - compared directly against this project's own native curve-tuft demo built the same way.)note")},
		{"F8", QObject::tr(R"note(Real curve geometry paired with Material "hair" for the first time in this project - exactly the combination that exposed a bug where the hair BSDF was using a flat-surface normal instead of the fiber's own tangent direction, which is what makes the highlight along each strand look like a strand instead of matte fuzz.)note")},
		{"F9", QObject::tr(R"note(pbrt's Shape "trianglemesh" "point2 uv" parameter - explicit per-vertex texture coordinates on a triangle mesh - threaded through both backends; GPU-recursive used to render this exact scene solid black before that plumbing existed.)note")},
		{"F10", QObject::tr(R"note(pbrt's PixelFilter directive selecting a box filter - every sample within a pixel counted equally, a hard cutoff at the pixel boundary - instead of the smoother default Gaussian. Compare the more aliased, harder-edged silhouettes here against any other scene's default-filtered render.)note")},

		// ---------------------------------------------------------------
		// Models (G25) - curated pbrt example
		// ---------------------------------------------------------------
		{"G25", QObject::tr(R"note(The classic pbrt-v4 "killeroo" statue example scene - a full end-to-end scene loaded from its own .pbrt file (geometry, materials, lights, camera all file-driven) rather than compiled directly into this renderer, the same file format and loading path a user's own custom pbrt scenes go through.)note")},
	};
	return kNotes;
}

// Returns the technique note for `sceneId`, or an honest fallback: an
// empty/unrecognized id (no scene selected, or the search box narrowed to
// zero matches) gets a neutral prompt; a real scene id with no authored
// entry - every scene outside the self-contained set - gets a message
// explaining why, rather than silently showing nothing or stale text.
inline QString forScene(const QString &sceneId) {
	if (sceneId.isEmpty())
		return QObject::tr("Select a scene to see the rendering technique it demonstrates.");
	const auto &map = notes();
	const auto it = map.constFind(sceneId);
	if (it != map.constEnd())
		return it.value();
	return QObject::tr(
		"No rendering-technique note is written for this scene yet - these are "
		"only authored for the self-contained scene set (the \"Self-Contained\" "
		"tab above) so far.");
}

// Debug-only drift guard, called once from createBasicTab() (mainwindow_tabs.cpp)
// after it enumerates the live scene list. scene_registry.h already lost one
// GUI-facing mirror table to exactly this failure mode once (see that file's
// own comment on scene_descriptor.h's old kScenes[]) and this table is the
// same shape of duplicate for a different field, so it gets the same kind of
// tripwire scene_registry_tests.cpp's GuiSceneCountMatchesRegistry test is
// for the scene count - just as a runtime qWarning() rather than a gtest
// assertion, because this header depends on Qt and the core test binary is
// deliberately Qt-free (see ray_tracer_tests.vcxproj's own comment on
// palette_file.h) so it can't link scene_registry.h itself to compare against
// (scene_metadata_client.h's comment: the GUI never links that header
// directly, only through the DLL boundary this function's caller already
// uses). Fails loud via qWarning, not a crash or a dialog - same "say so,
// don't stay silent" choice paneBackgroundRule()'s missing-resource warning
// makes elsewhere in this app.
inline void warnIfOutOfSync(const QStringList &selfContainedSceneIds) {
	const auto &map = notes();
	for (const QString &id : selfContainedSceneIds) {
		if (!map.contains(id)) {
			qWarning() << "scene_technique_notes.h: self-contained scene" << id
					   << "has no authored technique note - its info icon will show the generic fallback.";
		}
	}
	const QSet<QString> selfContainedSet(selfContainedSceneIds.constBegin(), selfContainedSceneIds.constEnd());
	for (auto it = map.constBegin(); it != map.constEnd(); ++it) {
		if (!selfContainedSet.contains(it.key())) {
			qWarning() << "scene_technique_notes.h: entry" << it.key()
					   << "does not match any current self-contained scene - stale, renamed, or moved to \"Requires External Files\"?";
		}
	}
}

} // namespace scene_technique_notes
