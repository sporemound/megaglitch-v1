"""Build and verify a Windows x64 portable MegaGlitch package from declared inputs.

Run with the CPython 3.12 interpreter selected by --python-root. There are no
downloads: installed dependency files, three standalone wheels, FFmpeg, and an
optional explicit GPU manifest are the build inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1]
IGNORE_NAMES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "target"}
STANDALONE = {"numpy": "2.2.6", "noise": "1.2.2", "opencv-python": "4.12.0.88"}
WHEEL_TAGS = {"numpy": "cp312-cp312-win_amd64", "noise": "cp312-cp312-win_amd64",
              "opencv-python": "cp37-abi3-win_amd64"}
LAUNCH_FILES = ("launch.py", "check.py", "render-picker.ps1", "runtime-env.cmd",
                "Start MegaGlitch.cmd", "Start MegaGlitch (CPU).cmd", "Start MegaGlitch (GPU).cmd",
                "Render Full-Frame.cmd", "Check Installation.cmd")
DOCUMENT_FILES = ("README.md", "README.txt", "VERIFICATION.md", "THIRD_PARTY_NOTICES.md", "SOURCE_PROVENANCE.json")
IGNORED_DRIVERS = {"nvcuda.dll", "nvencodeapi64.dll"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def ignore(directory, names):
    return [name for name in names if name in IGNORE_NAMES or name.endswith((".pyc", ".pyo"))
            or name == "MegaGlitchNode-Workstation.exe"]


def copy_path(source, destination):
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_lock(path):
    lock = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Expected a pinned name==version in {path.name}: {line}")
        name = normalized(match[1])
        if name in lock:
            raise ValueError(f"Duplicate dependency in {path.name}: {name}")
        lock[name] = match[2]
    for name, version in STANDALONE.items():
        if lock.get(name) != version:
            raise ValueError(f"The standalone {name} input must be pinned to {version}.")
    return lock


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--python-root", required=True, type=Path, help="Installed CPython 3.12 x64 prefix with the pinned CPU/GUI dependencies.")
    parser.add_argument("--ffmpeg-archive", required=True, type=Path, help="Windows static FFmpeg ZIP containing ffmpeg.exe, ffprobe.exe and its notices.")
    parser.add_argument("--output-root", required=True, type=Path, help="New portable package directory; its ZIP and SHA256 are written alongside it.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE, help="Source checkout containing app, preset and the root launch files.")
    parser.add_argument("--wheelhouse", type=Path, default=HERE / "wheels", help="Offline NumPy/noise CPython 3.12 wheels and the locked OpenCV CPU abi3 wheel, all Windows x64.")
    parser.add_argument("--gpu-plan", type=Path, help="Optional JSON manifest of existing GPU files and SHA256 values; never enabled by default.")
    parser.add_argument("--lock", type=Path, default=HERE / "requirements-runtime.lock", help="Pinned dependency lock.")
    parser.add_argument("--validate-inputs", action="store_true", help="Check input files and versions without assembling or running the package.")
    args = parser.parse_args()
    for name in ("python_root", "ffmpeg_archive", "output_root", "source_root", "wheelhouse", "lock", "gpu_plan"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def validate_host(args, lock):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Windows x64 CPython 3.12 from --python-root to run this builder.")
    if sys.maxsize <= 2 ** 32:
        raise RuntimeError("The portable runtime requires a 64-bit interpreter.")
    if Path(sys.executable).resolve().parent != args.python_root:
        raise RuntimeError("Run build.py with <python-root>/python.exe so dependency metadata matches the declared input.")
    for relative in ("python.exe", "Lib", "DLLs", "Library/bin/cairo.dll", "Library/lib/tcl8.6", "Library/lib/tk8.6"):
        if not (args.python_root / relative).exists():
            raise FileNotFoundError(f"Required Python/native input missing: {relative}")
    for relative in ("app/main.py", "app/effects_registry.py", "preset/render.py", "preset/node-adapter.py",
                     "preset/source-poster.png", "preset/soundtrack.wav", "LICENSE", "README.md", *LAUNCH_FILES):
        if not (args.source_root / relative).is_file():
            raise FileNotFoundError(f"Required repository input missing: {relative}")
    if not args.ffmpeg_archive.is_file():
        raise FileNotFoundError(args.ffmpeg_archive)
    site = args.python_root / "Lib/site-packages"
    available = {}
    for dist in metadata.distributions(path=[str(site)]):
        if dist.metadata.get("Name"):
            available.setdefault(normalized(dist.metadata["Name"]), []).append(dist)
    installed = {}
    from packaging.requirements import Requirement
    for name, expected in lock.items():
        if name in STANDALONE:
            continue
        matches = [dist for dist in available.get(name, ()) if dist.version == expected]
        if len(matches) != 1:
            actual = ", ".join(dist.version for dist in available.get(name, ())) or "not installed"
            raise RuntimeError(f"Expected one pinned dependency record: {name}=={expected}; declared Python has {actual}.")
        dist = matches[0]
        installed[name] = dist
        for requirement in dist.requires or ():
            req = Requirement(requirement)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            dependency = normalized(req.name)
            if dependency not in lock:
                raise RuntimeError(f"Lock omits active dependency {req.name} required by {name}.")
            if req.specifier and lock[dependency] not in req.specifier:
                raise RuntimeError(f"Lock version {dependency}=={lock[dependency]} does not satisfy {requirement}.")
    return installed


def wheel_inputs(wheelhouse):
    wheels = {}
    for name, version in STANDALONE.items():
        filename = f"{name.replace('-', '_')}-{version}-{WHEEL_TAGS[name]}.whl"
        matches = sorted(wheelhouse.glob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(f"Provide exactly one {filename} in --wheelhouse.")
        wheel = matches[0]
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                safe_member(member)
            records = [value for value in archive.namelist() if value.endswith(".dist-info/WHEEL")]
            identities = [value for value in archive.namelist() if value.endswith(".dist-info/METADATA")]
            if len(records) != 1 or len(identities) != 1:
                raise ValueError(f"Wheel has ambiguous metadata: {wheel.name}")
            if f"Tag: {WHEEL_TAGS[name]}" not in archive.read(records[0]).decode("utf-8"):
                raise ValueError(f"Wheel does not match the required compatible Windows x64 ABI: {wheel.name}")
            identity = archive.read(identities[0]).decode("utf-8")
            package_name = re.search(r"^Name: (.+)$", identity, re.M)
            package_version = re.search(r"^Version: (.+)$", identity, re.M)
            if not package_name or not package_version or normalized(package_name[1].strip()) != name or package_version[1].strip() != version:
                raise ValueError(f"Wheel metadata does not match its declared input name/version: {wheel.name}")
        wheels[name] = wheel
    return wheels


def safe_member(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"Unsafe archive/manifest path: {value}")
    return Path(*path.parts)


def extract_wheel(wheel, site, tag):
    with zipfile.ZipFile(wheel) as archive:
        wheel_records = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(wheel_records) != 1 or f"Tag: {tag}" not in archive.read(wheel_records[0]).decode("utf-8"):
            raise ValueError(f"Wheel tag differs from the declared compatible Windows x64 tag: {wheel.name}")
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = safe_member(member.filename)
            parts = relative.parts
            if parts and parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                    raise ValueError(f"Unsupported wheel relocation: {member.filename}")
                relative = Path(*parts[2:])
            destination = site / relative
            if destination.exists():
                raise FileExistsError(f"Standalone wheel would overwrite copied runtime files: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)


def copy_distributions(args, runtime, lock, installed):
    source_site = args.python_root / "Lib/site-packages"
    site = runtime / "Lib/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    included = []
    copied_names = set()
    for name, version in sorted(lock.items()):
        if name in STANDALONE:
            continue
        dist = installed[name]
        if not dist.files:
            raise RuntimeError(f"No install inventory for {name}; editable/unrecorded dependencies are not portable inputs.")
        top_names = {file.parts[0] for file in dist.files if file.parts and file.parts[0] not in (".", "..")}
        for top in sorted(top_names):
            if top in copied_names:
                continue
            source = source_site / top
            if source.exists():
                if not source.resolve().is_relative_to(args.python_root):
                    raise RuntimeError(f"Dependency escapes --python-root: {name}/{top}")
                copy_path(source, site / top)
                copied_names.add(top)
        included.append({"name": dist.metadata["Name"], "version": version})
    for name in ("scipy.libs", "pillow.libs"):
        if (source_site / name).exists():
            copy_path(source_site / name, site / name)
    return included


def runtime_core(args, runtime):
    runtime.mkdir()
    for source in args.python_root.iterdir():
        if source.is_file() and (source.suffix.lower() == ".dll" or source.name in ("python.exe", "pythonw.exe") or "license" in source.name.lower()):
            copy_path(source, runtime / source.name)
    def stdlib_ignore(directory, names):
        excluded = ignore(directory, names)
        if Path(directory) == args.python_root / "Lib":
            # CUDA's lowercase lib/ can share the Windows stdlib Lib directory.
            # Native Python modules live in DLLs; root Lib DLLs are not stdlib.
            excluded += ["site-packages"] + [name for name in names if Path(name).suffix.lower() == ".dll"]
        return excluded

    shutil.copytree(args.python_root / "Lib", runtime / "Lib", ignore=stdlib_ignore)
    copy_path(args.python_root / "DLLs", runtime / "DLLs")
    for name in ("tcl8", "tcl8.6", "tk8.6"):
        source = args.python_root / "Library/lib" / name
        if source.is_dir():
            copy_path(source, runtime / "Library/lib" / name)
    # Cairo is opened dynamically rather than imported by a PE dependency.
    copy_path(args.python_root / "Library/bin/cairo.dll", runtime / "Library/bin/cairo.dll")
    (runtime / "Lib/site-packages").mkdir(parents=True, exist_ok=True)
    (runtime / "Lib/site-packages/sitecustomize.py").write_text(
        '"""Keep DLL search handles alive in the portable runtime."""\n'
        'from pathlib import Path\nimport os\nimport sys\n'
        '_root = Path(sys.executable).resolve().parent\n_handles = []\n'
        'if hasattr(os, "add_dll_directory"):\n'
        '    for _path in (_root, _root / "DLLs", _root / "Library/bin"):\n'
        '        if _path.is_dir(): _handles.append(os.add_dll_directory(str(_path)))\n'
        'os.environ.setdefault("TCL_LIBRARY", str(_root / "Library/lib/tcl8.6"))\n'
        'os.environ.setdefault("TK_LIBRARY", str(_root / "Library/lib/tk8.6"))\n'
        'sys.dont_write_bytecode = True\n', encoding="utf-8")


def close_pe_dependencies(args, runtime):
    import pefile
    system = Path(os.environ["SystemRoot"]) / "System32"
    candidates = {}
    for directory in (args.python_root / "Library/bin", args.python_root / "DLLs", args.python_root):
        for source in directory.glob("*.dll"):
            candidates.setdefault(source.name.lower(), source)
    def is_amd64(path):
        binary = pefile.PE(str(path), fast_load=True)
        try:
            return binary.FILE_HEADER.Machine == 0x8664
        finally:
            binary.close()

    packaged = {}
    pending = []
    ignored_architectures = []
    for path in runtime.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".exe", ".pyd", ".dll"):
            continue
        if not is_amd64(path):
            ignored_architectures.append(path.relative_to(runtime).as_posix())
            continue
        pending.append(path)
        # EXE launcher templates are not DLL targets. Vendor-local DLL versions
        # may share a basename; each AMD64 variant still gets its imports checked.
        if path.suffix.lower() != ".exe":
            packaged.setdefault(path.name.lower(), []).append(path)
    checked, copied, unresolved, forwarders = set(), [], set(), set()
    while pending:
        path = pending.pop()
        if path in checked:
            continue
        checked.add(path)
        pe = pefile.PE(str(path), fast_load=True)
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY[name] for name in
                ("IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT", "IMAGE_DIRECTORY_ENTRY_EXPORT")])
            names = {entry.dll.decode("ascii") for entry in list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", ()))
                     + list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", ()))}
            exports = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
            for symbol in getattr(exports, "symbols", ()):
                if symbol.forwarder:
                    module = symbol.forwarder.decode("ascii").rsplit(".", 1)[0]
                    name = module if module.lower().endswith(".dll") else module + ".dll"
                    names.add(name)
                    forwarders.add(name.lower())
        finally:
            pe.close()
        for name in sorted(names):
            key = name.lower()
            if key.startswith(("api-ms-", "ext-ms-")) or key in IGNORED_DRIVERS:
                continue
            matches = packaged.get(key, ())
            # Resolve an owner's local library before the shared runtime dirs.
            local = (path.parent / name, runtime / name, runtime / "Library/bin" / name, runtime / "DLLs" / name)
            if any(candidate in matches for candidate in local) or matches:
                continue
            source = candidates.get(key)
            if source is not None and is_amd64(source):
                destination = runtime / "Library/bin" / source.name
                copy_path(source, destination)
                packaged.setdefault(key, []).append(destination)
                pending.append(destination)
                copied.append(source.relative_to(args.python_root).as_posix())
            elif not (system / name).is_file():
                unresolved.add(name)
    if unresolved:
        raise RuntimeError("Unresolved native dependencies: " + ", ".join(sorted(unresolved)))
    return {"added_dll_inputs": sorted(copied), "export_forwarder_modules": sorted(forwarders),
            "ignored_non_amd64_binaries": sorted(ignored_architectures),
            "scoped_native_variants": {name: sorted(p.relative_to(runtime).as_posix() for p in paths)
                                       for name, paths in packaged.items() if len(paths) > 1}, "unresolved": []}


def retain_native_licenses(args, runtime, package):
    # Match packaged binaries to conda file inventories, then retain the cached
    # package notices. Library inputs without conda metadata remain in the file
    # inventory and require notices in --python-root/Library/share/licenses.
    records = []
    inputs = [(path, json.loads(path.read_text(encoding="utf-8")))
              for path in sorted((args.python_root / "conda-meta").glob("*.json"))]
    for metadata_file, record in inputs:
        selected = False
        for relative in record.get("files", ()):
            # pip may replace site distributions without updating conda records.
            # Wheel metadata/notices cover those; conda owners cover native core.
            if relative.replace("\\", "/").lower().startswith("lib/site-packages/"):
                continue
            source = args.python_root / relative
            if not source.is_file() or source.suffix.lower() not in (".dll", ".pyd", ".exe"):
                continue
            matches = [runtime / relative, runtime / "Library/bin" / source.name]
            if any(destination.is_file() and sha(source) == sha(destination) for destination in matches):
                selected = True
                break
        if not selected:
            continue
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", f'{record.get("name", "package")}-{record.get("version", "unknown")}-{record.get("build", "unknown")}')
        target = package / "licenses/native" / label
        source_root = Path(record.get("link", {}).get("source", ""))
        license_root = source_root / "info/licenses"
        retained = False
        if source_root.is_dir() and license_root.is_dir():
            copy_path(license_root, target)
            retained = True
        license_file = source_root / "info/LICENSE.txt"
        if source_root.is_dir() and license_file.is_file():
            copy_path(license_file, target / "LICENSE.txt")
            retained = True
        about = source_root / "info/about.json"
        if source_root.is_dir() and about.is_file():
            copy_path(about, target / "about.json")
        donor = None
        if not retained and record.get("name") == "libfreetype6":
            # conda-forge splits the binaries and notices into two packages.
            donors = [(path, value) for path, value in inputs if value.get("name") == "freetype"
                      and value.get("version") == record.get("version")]
            for donor_path, donor_record in donors:
                notices = Path(donor_record.get("link", {}).get("source", "")) / "info/licenses"
                if notices.is_dir():
                    copy_path(notices, target / "freetype-notices")
                    donor = {key: donor_record.get(key) for key in ("name", "version", "build", "license", "sha256")}
                    donor["metadata_sha256"] = sha(donor_path)
                    retained = True
                    break
        declaration = str(record.get("license", "")).lower()
        declaration_only = not retained and declaration in ("blessing", "public domain", "public-domain", "unlicense")
        if declaration_only:
            retained = True
        entry = {key: record.get(key) for key in ("name", "version", "build", "license", "sha256")}
        entry.update({"metadata_sha256": sha(metadata_file), "notices_retained": retained,
                      "notice_donor": donor, "license_declaration_only": declaration_only})
        write_json(target / "PACKAGE.json", entry)
        records.append(entry)
    for relative in ("Library/share/licenses", "info/licenses"):
        source = args.python_root / relative
        if source.is_dir():
            copy_path(source, package / "licenses/python-input" / Path(relative).name)
    missing = [record["name"] for record in records if not record["notices_retained"]]
    if missing:
        raise RuntimeError("Native package notices unavailable in the declared Python input/cache: " + ", ".join(missing))
    return records


def extract_ffmpeg(archive_path, package):
    found = {}
    notices = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = safe_member(member.filename)
            if relative.name in ("ffmpeg.exe", "ffprobe.exe"):
                if relative.name in found:
                    raise ValueError(f"FFmpeg archive has duplicate {relative.name} entries.")
                destination = package / "tools" / relative.name
                found[relative.name] = member.filename
            else:
                destination = package / "tools/ffmpeg-distribution" / relative
                if "license" in relative.name.lower() or "copying" in relative.name.lower():
                    notices.append(relative.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    if set(found) != {"ffmpeg.exe", "ffprobe.exe"}:
        raise ValueError("The FFmpeg archive must contain both ffmpeg.exe and ffprobe.exe.")
    if not notices:
        raise ValueError("The FFmpeg archive must include its license/notices.")
    return {"archive": archive_path.name, "sha256": sha(archive_path), "binaries": found, "notices": notices}


def validate_ffmpeg_archive(path):
    found, notices = [], []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = safe_member(member.filename)
            if member.is_dir():
                continue
            if relative.name in ("ffmpeg.exe", "ffprobe.exe"):
                found.append(relative.name)
            if "license" in relative.name.lower() or "copying" in relative.name.lower():
                notices.append(relative.name)
    if sorted(found) != ["ffmpeg.exe", "ffprobe.exe"] or not notices:
        raise ValueError("FFmpeg ZIP must contain exactly one ffmpeg.exe, one ffprobe.exe and their license/notices.")


def gpu_inputs(plan_path):
    if plan_path is None:
        return None
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = Path(plan["source_root"])
    if not source.is_absolute():
        source = plan_path.parent / source
    source = source.resolve()
    files = plan.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("GPU plan must declare a nonempty files map of relative path: SHA256.")
    for value, expected in files.items():
        relative = safe_member(value)
        path = source / relative
        if not path.resolve().is_relative_to(source) or not path.is_file() or sha(path) != expected.lower():
            raise ValueError(f"GPU file is missing or differs from its declared SHA256: {value}")
    for directory in ("python", "opencv", "cuda"):
        if not any(safe_member(name).parts[0] == directory for name in files):
            raise ValueError(f"GPU plan lacks the required {directory}/ files.")
    if not any("license" in name.lower() or "copying" in name.lower() for name in files):
        raise ValueError("GPU plan must also include the notices for its redistributed components.")
    return source, files, sha(plan_path)


def isolated_environment(package):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV", "MEGAGLITCH_NODE_ROOT",
                 "MEGAGLITCH_OPENCV_BIN", "MEGAGLITCH_OPENCV_PYTHON", "MEGAGLITCH_CUDA_PATH", "MEGAGLITCH_CUDA_BIN"):
        env.pop(name, None)
    windows_root = env.get("SYSTEMROOT") or env.get("SystemRoot") or env.get("WINDIR")
    if not windows_root:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_windows_directory = kernel32.GetWindowsDirectoryW
        get_windows_directory.argtypes = (ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint)
        get_windows_directory.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_windows_directory(buffer, len(buffer))
        if length == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        if length >= len(buffer):
            raise OSError("GetWindowsDirectoryW returned a directory longer than its buffer.")
        windows_root = buffer.value
    windows = Path(windows_root)
    for name in list(env):
        if name.upper() in ("SYSTEMROOT", "WINDIR"):
            env.pop(name)
    env.update({"SYSTEMROOT": str(windows), "WINDIR": str(windows),
        "PATH": os.pathsep.join(map(str, (package / "runtime", package / "runtime/Library/bin", package / "tools", windows / "System32", windows))),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "TEMP": str(package / "temp"), "TMP": str(package / "temp"),
        "TCL_LIBRARY": str(package / "runtime/Library/lib/tcl8.6"), "TK_LIBRARY": str(package / "runtime/Library/lib/tk8.6"),
        "IMAGEIO_FFMPEG_EXE": str(package / "tools/ffmpeg.exe")})
    return env


def run_check(command, package, log_name, timeout=300):
    log = package / "verification" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Verifying " + log_name + "...", flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, env=isolated_environment(package), cwd=package, stdout=output, stderr=subprocess.STDOUT,
            timeout=timeout, creationflags=0x08000000 if sys.platform == "win32" else 0)
    if result.returncode:
        raise RuntimeError(f"Portable verification failed ({result.returncode}); see {log}")


def verify_package(package):
    python = package / "runtime/python.exe"
    run_check([str(python), "-I", str(package / "check.py")], package, "dependency-check.log")
    output = package / "verification/portable-4k-smoke.mp4"
    run_check([str(python), "-I", str(package / "preset/render.py"), "--seconds", "2", "--encoder", "auto", "--output", str(output)],
              package, "4k-export.log", timeout=900)
    probe = subprocess.check_output([str(package / "tools/ffprobe.exe"), "-v", "error", "-show_entries",
        "stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels:format=duration,size", "-of", "json", str(output)],
        env=isolated_environment(package), cwd=package, text=True,
        creationflags=0x08000000 if sys.platform == "win32" else 0)
    info = json.loads(probe)
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    if (video["codec_name"] != "h264" or (video["width"], video["height"]) != (2160, 3840)
            or video["r_frame_rate"] != "24/1" or int(video["nb_frames"]) != 48
            or abs(float(info["format"]["duration"]) - 2.0) > 0.01
            or audio["codec_name"] != "aac" or audio["channels"] != 2 or int(audio["sample_rate"]) != 48000):
        raise RuntimeError("Portable 4K export failed its media contract: " + probe)
    write_json(package / "verification/portable-export-check.json", info)
    return {"dependency_imports": True, "full_4k_export": True, "clean_machine_certification": False,
            "editor_effects": "CPU", "export_encoder": "auto (NVIDIA NVENC when available, otherwise CPU libx264)"}


def inventory(package):
    files = sorted(path for path in package.rglob("*") if path.is_file() and path.name != "BUNDLE-SHA256.txt")
    lines = [f"{sha(path)}  {path.relative_to(package).as_posix()}" for path in files]
    (package / "BUNDLE-SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files) + 1


def archive_package(package):
    destination = package.parent / (package.name + ".zip")
    checksum = destination.with_suffix(".zip.sha256")
    if destination.exists() or checksum.exists():
        raise FileExistsError("Existing download or checksum preserved; choose a new --output-root.")
    files = sorted(path for path in package.rglob("*") if path.is_file())
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path in files:
            entry = zipfile.ZipInfo(package.name + "/" + path.relative_to(package).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            with path.open("rb") as source, archive.open(entry, "w", force_zip64=True) as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity check failed: {bad}")
        archived = {member.filename for member in archive.infolist() if not member.is_dir()}
    expected = {package.name + "/" + path.relative_to(package).as_posix() for path in files}
    if archived != expected:
        raise RuntimeError("ZIP inventory does not match the verified package.")
    digest = sha(destination)
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return {"archive": destination.name, "sha256": digest, "bytes": destination.stat().st_size,
            "zip_integrity": True, "archive_inventory_matches": True, "files": len(files)}


def main():
    args = arguments()
    lock = load_lock(args.lock)
    installed = validate_host(args, lock)
    wheels = wheel_inputs(args.wheelhouse)
    validate_ffmpeg_archive(args.ffmpeg_archive)
    gpu = gpu_inputs(args.gpu_plan)
    for protected in (args.python_root, args.source_root / "app", args.source_root / "preset"):
        if args.output_root.is_relative_to(protected):
            raise ValueError("--output-root must be outside the Python input and application/preset source trees.")
    archive = args.output_root.parent / (args.output_root.name + ".zip")
    if args.output_root.exists() or archive.exists() or archive.with_suffix(".zip.sha256").exists():
        raise FileExistsError("Existing package/download preserved; choose a new --output-root.")
    if args.validate_inputs:
        print("Build input checks passed. No package was assembled.")
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    package = args.output_root
    for name in ("outputs", "temp", "verification", "licenses"):
        (package / name).mkdir()
    print("Copying repository application, preset and launch files...", flush=True)
    copy_path(args.source_root / "app", package / "app")
    copy_path(args.source_root / "preset", package / "preset")
    for name in (*LAUNCH_FILES, "LICENSE"):
        copy_path(args.source_root / name, package / name)
    for name in DOCUMENT_FILES:
        if (args.source_root / name).is_file():
            copy_path(args.source_root / name, package / name)
    copy_path(args.lock, package / "BUILD-INPUTS/requirements-runtime.lock")
    runtime = package / "runtime"
    print("Copying CPython core and pinned installed dependencies...", flush=True)
    runtime_core(args, runtime)
    included = copy_distributions(args, runtime, lock, installed)
    for name, wheel in wheels.items():
        extract_wheel(wheel, runtime / "Lib/site-packages", WHEEL_TAGS[name])
        included.append({"name": name, "version": STANDALONE[name]})
    native = close_pe_dependencies(args, runtime)
    licenses = retain_native_licenses(args, runtime, package)
    ffmpeg = extract_ffmpeg(args.ffmpeg_archive, package)
    if gpu is not None:
        gpu_root, gpu_files, gpu_hash = gpu
        for relative in sorted(gpu_files):
            copy_path(gpu_root / safe_member(relative), package / "gpu" / safe_member(relative))
        write_json(package / "gpu/BUILD-INPUTS.json", {"plan_sha256": gpu_hash, "files": gpu_files,
            "enabled_by_default": False, "verification": "Declared existing files verified by SHA256; GPU editor requires an explicit launch and a compatible NVIDIA driver."})
    print("Checking portable imports and rendering a real two-second 4K export...", flush=True)
    verification = verify_package(package)
    info = {"package": "MegaGlitch Full-Frame Windows", "builder": "packaging/portable/build.py",
        "python": sys.version, "python_executable_sha256": sha(args.python_root / "python.exe"),
        "source_files_modified": False, "default_editor_backend": "CPU", "optional_gpu": gpu is not None,
        "dependency_lock_sha256": sha(args.lock), "distributions": sorted(included, key=lambda value: normalized(value["name"])),
        "standalone_wheels": {name: {"file": wheel.name, "sha256": sha(wheel)} for name, wheel in sorted(wheels.items())},
        "native_library_closure": native, "native_license_records": licenses, "ffmpeg": ffmpeg,
        "verification": verification, "reproducibility": "Offline declared-input build. BUNDLE-SHA256 records every output file; runtime/core/native inputs are preserved as copied binary files. Verification timing and hardware encoder output may differ between machines."}
    write_json(package / "BUILD-INFO.json", info)
    inventory(package)
    result = archive_package(package)
    write_json(package.parent / (package.name + "-build-result.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Build failed: {error}", file=sys.stderr)
        sys.exit(1)
