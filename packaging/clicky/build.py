"""Build a movable macOS app using the project's CPython 3.12 runtime and hashed locks."""

import argparse
import ctypes
import os
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import AppKit
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "clicky"
IDENTIFIER = "local.clicky.assistant"


def run(*argv):
    subprocess.run(argv, check=True)


def make_icon(temp, resources):
    iconset = temp / "Clicky.iconset"
    iconset.mkdir()
    icon = Image.new("RGBA", (1024, 1024))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle((65, 70, 959, 964), radius=196, fill="#262b31", outline="#41464d", width=12)
    nib = [(280, 246), (782, 443), (556, 540), (468, 772)]
    draw.polygon(nib, fill="#ffb347")
    draw.polygon([nib[0], nib[2], nib[3]], fill="#e68a2c")
    draw.line([*nib, nib[0]], fill="#ffe7bb", width=8, joint="curve")
    for size in (16, 32, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            pixels = size * scale
            icon.resize((pixels, pixels), Image.Resampling.LANCZOS).save(
                iconset / f"icon_{size}x{size}{suffix}.png"
            )
    run("iconutil", "-c", "icns", iconset, "-o", resources / "Clicky.icns")


def build(destination, local):
    if destination.exists():
        info_path = destination / "Contents/Info.plist"
        if (
            destination.is_symlink()
            or not info_path.is_file()
            or plistlib.loads(info_path.read_bytes()).get("CFBundleIdentifier") != IDENTIFIER
        ):
            raise SystemExit("An unrelated item occupies the output path. Choose a different --output.")
        running = AppKit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(IDENTIFIER)
        if any(app.bundleURL() and Path(app.bundleURL().path()).resolve() == destination for app in running):
            raise SystemExit("Quit this copy of Clicky before rebuilding it, or choose a new --output.")
    base = Path(sys.base_prefix)
    library = base / "lib" / "libpython3.12.dylib"
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 12) or not library.is_file():
        raise SystemExit("Build with the uv-managed macOS CPython 3.12 used by Clicky.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        bundle = Path(temp) / "Clicky.app"
        contents = bundle / "Contents"
        runtime = contents / "Resources/runtime"
        stdlib = runtime / "lib/python3.12"
        frameworks = contents / "Frameworks"
        binary = contents / "MacOS/Clicky"
        frameworks.mkdir(parents=True)
        binary.parent.mkdir()
        shutil.copytree(
            base / "lib/python3.12",
            stdlib,
            ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc", "config-3.12-*"),
        )
        library_copy = frameworks / library.name
        shutil.copy2(library, library_copy)
        install_name = "@executable_path/../Frameworks/" + library.name
        run("install_name_tool", "-id", install_name, library_copy)
        run("codesign", "--force", "--sign", "-", library_copy)
        run(
            "clang",
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-I",
            sysconfig.get_path("include"),
            Path(__file__).with_name("launcher.m"),
            library_copy,
            "-o",
            binary,
        )
        site = stdlib / "site-packages"
        lock = SOURCE / ("requirements-local.lock" if local else "requirements.lock")
        run(
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            site,
            "--require-hashes",
            "-r",
            lock,
        )
        for path in (*SOURCE.rglob("*.py"), SOURCE / "config.yaml"):
            installed = site / path.relative_to(ROOT)
            installed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, installed)
        shutil.copy2(ROOT / "LICENSE", contents / "Resources/Clicky-LICENSE")
        python_license = base / "lib/python3.12/LICENSE.txt"
        if not python_license.is_file():
            raise SystemExit("The CPython license file is missing from the runtime.")
        make_icon(Path(temp), contents / "Resources")
        info = {
            "CFBundleIconFile": "Clicky.icns",
            "CFBundleIdentifier": IDENTIFIER,
            "CFBundleExecutable": "Clicky",
            "CFBundleName": "CursorMix",
            "CFBundleDisplayName": "CursorMix",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "0.1",
            "CFBundleVersion": "1",
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Record speech while you hold the CursorMix shortcut.",
            "NSScreenCaptureUsageDescription": "See the screen you summon CursorMix on to answer and guide you.",
        }
        (contents / "Info.plist").write_bytes(plistlib.dumps(info))
        run("codesign", "--force", "--deep", "--sign", "-", bundle)
        run("codesign", "--verify", "--deep", "--strict", bundle)
        if destination.exists():
            # macOS renamex_np(RENAME_SWAP) publishes a complete replacement in a single operation.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.renamex_np(os.fsencode(bundle), os.fsencode(destination), 0x00000002):
                raise OSError(ctypes.get_errno(), "Could not replace the previous Clicky build")
        else:
            os.rename(bundle, destination)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Clicky.app")
    parser.add_argument(
        "--local", action="store_true", help="include the local vision and voice dependencies"
    )
    parser.add_argument(
        "--use-checkout-env",
        action="store_true",
        help="save the path of this checkout's .env.local in your macOS preferences",
    )
    args = parser.parse_args()
    build(args.output.absolute(), args.local)
    if args.use_checkout_env:
        path = ROOT / ".env.local"
        if not path.is_file():
            raise SystemExit("The bundle was built. This checkout has no .env.local to select.")
        run("defaults", "write", IDENTIFIER, "CheckoutEnvironmentFile", "-string", path)
