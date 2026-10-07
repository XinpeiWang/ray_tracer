#!/usr/bin/env python3
"""Generates qt_gui/scene_packs.txt: the catalogue behind the GUI's "Download missing files" for the scenes whose files are NOT in
this repository - the large third-party environment scenes (Sponza, Bistro, San Miguel, ...) and the pbrt-v4-scenes examples
(Zero Day, Villa, ...). Everything is fetched from the ORIGINAL upstream sites, directly to the user's machine; this project hosts and
redistributes none of it.

Run from the repo root (needs network, python 3.8+, no third-party packages):
    python3 scripts/gen_scene_packs.py            # rewrites qt_gui/scene_packs.txt
    python3 scripts/gen_scene_packs.py --check    # exits 1 if a pack no longer resolves (an upstream file moved or changed)

How the data is gathered, and why the app never downloads a whole archive it does not need:
  * McGuire Computer Graphics Archive zips: the zip's CENTRAL DIRECTORY is read with HTTP range requests (a few KB), which gives every
    member's size, CRC-32 and header offset. The app later fetches only the members a scene references, again with range requests
    (Bistro's textures are two ~0.5 GB zips of which a scene uses ~100 MB). Each extracted file is checked against its recorded CRC-32;
    an archive whose size no longer matches the recorded one is refused ("upstream changed").
  * pbrt-v4-scenes: the repository tree at one pinned commit lists every file with its size and git blob SHA-1; the app fetches each
    file from raw.githubusercontent.com at that commit and verifies the blob SHA-1, so the content is exactly what was pinned.
  * Which files a McGuire-based scene needs is read from the scene's own pbrt file in pbrt_scenes/ (every "string filename"), so the
    catalogue cannot drift from what the scene really loads.

Manifest format (one record per line, fields separated by '|', '#' starts a comment):
    pack|<id>|<title>|<credit and licence>|<source page url>
    archive|<url>|<archive size in bytes>
    member|<dest path under the app/user folder>|<member name in the archive>|<compressed>|<uncompressed>|<crc32 hex>|<local header offset>|<method>|<name len>|<extra len>
    file|<url>|<dest path>|<size>|<git blob sha1>
A "member" belongs to the most recent "archive", which belongs to the most recent "pack"; the same for "file".
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "qt_gui", "scene_packs.txt")
UA = {"User-Agent": "RayTracerGUI"}   # the McGuire archive's server answers 406 to a default python User-Agent
G3D = "https://casual-effects.com/g3d/data10/"
PBRT_REPO = "mmp/pbrt-v4-scenes"


class HttpFile(io.RawIOBase):
    """A read-only seekable view of a remote file via HTTP range requests, so zipfile can read just the central directory."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as f:
            self.size = int(f.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size - 1, self.pos + n - 1)
        req = urllib.request.Request(self.url, headers=dict(UA, Range="bytes=%d-%d" % (self.pos, end)))
        with urllib.request.urlopen(req, timeout=120) as f: data = f.read()
        self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def referenced_files(scene_file):
    """Every file a pbrt scene names with "string filename", as a path relative to the repo root (after resolving ../), '#group' dropped."""
    text = open(os.path.join(REPO, "pbrt_scenes", scene_file), encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r'"string filename"\s*\[?\s*"([^"]+)"', text):
        rel = m.group(1).split("#")[0]
        norm = os.path.normpath(os.path.join("pbrt_scenes", rel)).replace("\\", "/")
        if norm not in out: out.append(norm)
    return out


# --- McGuire archive packs ---------------------------------------------------------------------------------------------------------
# Each pack: the scene's pbrt file (what it needs is read from it), the archives (url, dest prefix for non-obj members, {obj member: dest}),
# and the credit/licence text shown before anything is downloaded.
MCGUIRE = [
    dict(id="sponza", title="Crytek Sponza", scene="environment-sponza.pbrt",
         archives=[(G3D + "common/model/crytek_sponza/sponza.zip", "models/sponza_textures/", {"sponza.obj": "models/sponza.obj"})]),
    dict(id="bistro", title="Amazon Lumberyard Bistro (exterior)", scene="environment-bistro-exterior.pbrt",
         archives=[(G3D + "research/model/bistro/Exterior.zip", "models/bistro_textures/Exterior/", {"exterior.obj": "models/bistro_exterior.obj"}),
                   (G3D + "research/model/bistro/BuildingTextures", "models/bistro_textures/BuildingTextures/", {}),
                   (G3D + "research/model/bistro/OtherTextures", "models/bistro_textures/OtherTextures/", {}),
                   (G3D + "research/model/bistro/PropTextures", "models/bistro_textures/PropTextures/", {})]),
    dict(id="rungholt", title="Rungholt", scene="environment-rungholt.pbrt",
         archives=[(G3D + "research/model/rungholt/rungholt.zip", "models/rungholt_textures/", {"rungholt.obj": "models/rungholt.obj"})]),
    dict(id="fireplace-room", title="Fireplace Room", scene="environment-fireplace-room.pbrt",
         archives=[(G3D + "research/model/fireplace_room/fireplace_room.zip", "models/fireplace_room_textures/", {"fireplace_room.obj": "models/fireplace_room.obj"})]),
    dict(id="san-miguel", title="San Miguel 2.0", scene="environment-san-miguel.pbrt",
         archives=[(G3D + "research/model/San_Miguel/San_Miguel.zip", "models/san_miguel_textures/", {"san-miguel.obj": "models/san_miguel.obj"})]),
    dict(id="sibenik", title="Sibenik Cathedral", scene="environment-sibenik-cathedral.pbrt",
         archives=[(G3D + "research/model/sibenik/sibenik.zip", "models/sibenik_cathedral_textures/", {"sibenik.obj": "models/sibenik_cathedral.obj"})]),
    dict(id="breakfast-room", title="Breakfast Room", scene="environment-breakfast-room.pbrt",
         archives=[(G3D + "research/model/breakfast_room/breakfast_room.zip", "models/breakfast_room_textures/", {"breakfast_room.obj": "models/breakfast_room.obj"})]),
    dict(id="salle-de-bain", title="Salle de Bain", scene="environment-salle-de-bain.pbrt",
         archives=[(G3D + "research/model/salle_de_bain/salle_de_bain.zip", "models/salle_de_bain_textures/", {"salle_de_bain.obj": "models/salle_de_bain.obj"})]),
    dict(id="gallery", title="Gallery", scene="environment-gallery.pbrt",
         archives=[(G3D + "research/model/gallery/gallery.zip", "models/gallery_textures/", {"gallery.obj": "models/gallery.obj"})]),
    dict(id="lost-empire", title="Lost Empire", scene="environment-lost-empire.pbrt",
         archives=[(G3D + "research/model/lost_empire/lost-empire.zip", "models/lost_empire_textures/", {"lost_empire.obj": "models/lost_empire.obj"})]),
    dict(id="vokselia-spawn", title="Vokselia Spawn", scene="environment-vokselia-spawn.pbrt",
         archives=[(G3D + "research/model/vokselia_spawn/vokselia_spawn.zip", "models/vokselia_spawn_textures/", {"vokselia_spawn.obj": "models/vokselia_spawn.obj"})]),
    dict(id="power-plant", title="Power Plant", scene="environment-power-plant.pbrt",
         note="Released for NON-COMMERCIAL use only - by downloading you accept those terms (http://gamma.cs.unc.edu/POWERPLANT/).",
         archives=[(G3D + "research/model/powerplant/powerplant.zip", "models/powerplant_textures/", {"powerplant.obj": "models/powerplant.obj"})]),
]

# --- pbrt-v4-scenes folders ---------------------------------------------------------------------------------------------------------
# folder -> (title, credit/licence). The scene's .pbrt files and everything they use live in the folder; all of it is fetched.
PBRT_V4 = {
    "contemporary-bathroom": ("Contemporary Bathroom", "Model by Mareck via Blend Swap, CC0 (public domain). From mmp/pbrt-v4-scenes."),
    "barcelona-pavilion": ("Barcelona Pavilion", "Model by Hamza Cheggour (emirage.org), CC BY 2.0. From mmp/pbrt-v4-scenes."),
    "sssdragon": ("Subsurface Dragon", "Dragon courtesy of the Stanford Computer Graphics Laboratory; environment map by Bernhard Vogl. From mmp/pbrt-v4-scenes."),
    "ganesha": ("Ganesha", "Statue scanned by Wenzel Jakob. From mmp/pbrt-v4-scenes."),
    "sportscar": ("Sports Car", "Model and pbrt conversion by Yasutoshi Mori (@MirageYM), CC BY 2.0; measured BRDFs from the RGL Material Database. From mmp/pbrt-v4-scenes."),
    "zero-day": ("Zero Day", "Frames converted from Beeple's Zero-Day; see http://beeple-crap.com/resources.php for his licence. From mmp/pbrt-v4-scenes."),
    "crown": ("Crown", "Model by Martin Lubich (http://www.loramel.net/). From mmp/pbrt-v4-scenes."),
    "villa": ("Villa", "Scene by Florent Boyer. From mmp/pbrt-v4-scenes."),
    "transparent-machines": ("Transparent Machines", "Models from frames of Beeple's Transparent Machines; see http://beeple-crap.com/resources.php for his licence. From mmp/pbrt-v4-scenes."),
}


def quote_path(p):
    return "/".join(urllib.parse.quote(seg) for seg in p.split("/"))


def strip_tags(html):
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&copy;", "(c)").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def mcguire_credit(spec):
    """Copyright and licence exactly as the archive's own info.js states them (not retyped here), plus any extra terms."""
    url = spec["archives"][0][0]
    info_url = url.rsplit("/", 1)[0] + "/info.js"
    with urllib.request.urlopen(urllib.request.Request(info_url, headers=UA), timeout=60) as f:
        info = f.read().decode("utf-8", "replace")
    fields = {}
    for key in ("copyright", "license"):
        m = re.search(key + r':\s*"((?:[^"\\]|\\.)*)"', info)
        fields[key] = strip_tags(m.group(1)) if m else "see the archive page"
    text = "%s. License: %s. From the McGuire Computer Graphics Archive." % (fields["copyright"], fields["license"])
    if spec.get("note"): text += " " + spec["note"]
    return text


def mcguire_pack(spec, problems):
    lines = ["pack|%s|%s|%s|https://casual-effects.com/data/" % (spec["id"], spec["title"], mcguire_credit(spec))]
    needed = referenced_files(spec["scene"])
    unresolved = [d for d in needed]
    total_dl = total_disk = 0
    for url, tex_prefix, obj_map in spec["archives"]:
        zf_file = HttpFile(url)
        zf = zipfile.ZipFile(zf_file)
        members = {i.filename.lower(): i for i in zf.infolist() if not i.is_dir()}
        used = []
        for dest in list(unresolved):
            member = None
            for name, d in obj_map.items():
                if d == dest: member = name
            if member is None and dest.startswith(tex_prefix):
                member = dest[len(tex_prefix):]
            if member is None: continue
            info = members.get(member.lower())
            if info is None:
                # bistro-exterior.pbrt names "Metal_ RollDoor_01" (a stray space from the converter that wrote it); upstream has "Metal_RollDoor_01".
                # The dest keeps the scene's spelling, so the scene finds it; only the archive lookup is lenient.
                info = members.get(member.replace("_ ", "_").lower())
            if info is None: continue
            used.append((dest, info))
            unresolved.remove(dest)
        if not used: continue
        lines.append("archive|%s|%d" % (url, zf_file.size))
        for dest, i in used:
            nlen = len(i.orig_filename.encode("utf-8"))
            lines.append("member|%s|%s|%d|%d|%08x|%d|%d|%d|%d" % (dest, i.filename, i.compress_size, i.file_size, i.CRC, i.header_offset,
                                                              i.compress_type, nlen, len(i.extra)))
            total_dl += i.compress_size; total_disk += i.file_size
    if unresolved:
        problems.append("%s: %d referenced file(s) not found in any archive, e.g. %s" % (spec["id"], len(unresolved), unresolved[:3]))
    print("%-16s %4d files, download %8.1f MB, on disk %8.1f MB" % (spec["id"], len(needed) - len(unresolved), total_dl / 1e6, total_disk / 1e6))
    return lines


def outside_references(folder, tree, commit, blobs, by_path):
    """Files in OTHER folders of the repository that this folder's .pbrt files name (barcelona-pavilion borrows trees from ../landscape).
    Quoted "../..." strings are resolved against the file that names them; only paths that exist in the tree are kept. Scanned to a fixed point,
    since a borrowed .pbrt can itself name more files."""
    extra, queue, seen = [], [e["path"] for e in blobs if e["path"].endswith(".pbrt")], set()
    while queue:
        path = queue.pop()
        if path in seen: continue
        seen.add(path)
        url = "https://raw.githubusercontent.com/%s/%s/%s" % (PBRT_REPO, commit, quote_path(path))
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as f:
            text = f.read().decode("utf-8", errors="replace")
        for ref in re.findall(r'"((?:\.\./)[^"]+)"', text):
            norm = os.path.normpath(os.path.join(os.path.dirname(path), ref.split("#")[0])).replace("\\", "/")
            e = by_path.get(norm)
            if e is None or norm.startswith(folder + "/") or e in extra: continue
            extra.append(e)
            if norm.endswith(".pbrt"): queue.append(norm)
    return extra


def pbrt_v4_packs(problems):
    base = "https://api.github.com/repos/%s" % PBRT_REPO
    with urllib.request.urlopen(urllib.request.Request(base + "/commits/master", headers=UA), timeout=60) as f:
        commit = json.load(f)["sha"]
    with urllib.request.urlopen(urllib.request.Request(base + "/git/trees/%s?recursive=1" % commit, headers=UA), timeout=120) as f:
        tree = json.load(f)
    if tree.get("truncated"): problems.append("pbrt-v4-scenes tree listing was truncated")
    by_path = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    lines = ["# pbrt-v4-scenes pinned at commit %s" % commit]
    for folder, (title, credit) in PBRT_V4.items():
        blobs = [e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith(folder + "/")]
        if not blobs: problems.append("pbrt-v4-scenes has no folder %s" % folder); continue
        blobs += outside_references(folder, tree, commit, blobs, by_path)
        lines.append("pack|%s|%s|%s|https://github.com/%s/tree/master/%s" % (folder, title, credit, PBRT_REPO, folder))
        for e in blobs:
            lines.append("file|https://raw.githubusercontent.com/%s/%s/%s|pbrt_scenes/%s|%d|%s" % (PBRT_REPO, commit, quote_path(e["path"]), e["path"], e["size"], e["sha"]))
        total = sum(e["size"] for e in blobs)
        print("%-22s %5d files, %8.1f MB" % (folder, len(blobs), total / 1e6))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the generated catalogue differs from the committed one")
    args = ap.parse_args()
    problems = []
    out = ["# Generated by scripts/gen_scene_packs.py - do not edit by hand. See that script for the format and the reasoning.",
           "# Third-party content, fetched from its original sites on the user's request; nothing here is hosted or redistributed by this project."]
    for spec in MCGUIRE:
        out += mcguire_pack(spec, problems)
    out += pbrt_v4_packs(problems)
    if problems:
        print("\nPROBLEMS:", *problems, sep="\n  ")
        sys.exit(1)
    text = "\n".join(out) + "\n"
    if args.check:
        same = os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == text
        print("catalogue is up to date" if same else "catalogue DIFFERS from the committed one - rerun without --check")
        sys.exit(0 if same else 1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f: f.write(text)
    print("wrote", OUT, "(%d lines)" % len(out))


if __name__ == "__main__":
    main()
