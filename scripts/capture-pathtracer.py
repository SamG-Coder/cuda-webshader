"""Validate the original native PPM and retain a lossless PNG and provenance."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess
import zlib

root = Path(__file__).resolve().parents[1]
upstream = root / '.local/raytracing-cuda'
revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != 'ab140b12d4923b75270831baabab5e4d4209f305':
    raise ValueError('Unexpected path tracer revision')
if subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only'], text=True).strip():
    raise ValueError('Upstream tracked source was modified')
ppm = (root / '.local/pathtracer-native.ppm').read_bytes()
tokens = ppm.split()
if tokens[:4] != [b'P3', b'1200', b'800', b'255']:
    raise ValueError('Unexpected native image dimensions or format')
values = list(map(int, tokens[4:]))
if len(values) != 1200 * 800 * 3 or min(values) < 0 or max(values) > 255:
    raise ValueError('Incomplete native image or invalid colour range')
pixels = bytes(values)

def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)

scan = b''.join(b'\0' + pixels[y*3600:(y+1)*3600] for y in range(800))
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1200, 800, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(scan)) + chunk(b'IEND', b'')
(root / 'reports/pathtracer-native.png').write_bytes(png)
log = (root / 'reports/pathtracer-native.txt').read_text()
seconds = float(re.search(r'took ([0-9.]+) seconds', log)[1])
files = ['main.cu', 'vec3.h', 'ray.h', 'hitable.h', 'hitable_list.h', 'sphere.h', 'camera.h', 'material.h']
manifest = {
    'repository': 'https://github.com/rogerallen/raytracinginoneweekendincuda',
    'revision': revision,
    'sourceUnchanged': True,
    'sourceSha256LF': {name: hashlib.sha256((upstream/name).read_text().replace('\r\n', '\n').encode()).hexdigest() for name in files},
    'width': 1200, 'height': 800, 'samplesPerPixel': 10, 'maximumBounces': 50, 'spheres': 488,
    'nativeReportedSeconds': seconds,
    'timingScope': 'Original timer: per-pixel RNG initialization and render, including synchronization; excludes world construction and PPM output.',
    'buildFlags': ['-O3', '-std=c++17', '-arch=native'],
    'rgbSha256': hashlib.sha256(pixels).hexdigest(),
    'ppmSha256': hashlib.sha256(ppm).hexdigest(),
    'pngSha256': hashlib.sha256(png).hexdigest(),
    'validatedPixels': 960000,
    'webgpuVerified': False,
}
(root/'reports/pathtracer-native-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print(f'Validated {manifest["validatedPixels"]} native pixels; original reported time {seconds}s')
