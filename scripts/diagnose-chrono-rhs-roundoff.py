"""Independent float64 diagnostics for the initialized, zero-velocity dam break.
This is validation only; no values from this script are used in GPU execution.
"""
import json, math, struct
from pathlib import Path
n, total = 30327, 794643
raw = Path('reports/chrono-adami-input.bin').read_bytes()
offset = 0
def take(fmt, count):
    global offset
    size = struct.calcsize(fmt)
    values = list(struct.iter_unpack(fmt, raw[offset:offset+size*count]))
    offset += size*count
    return values
offsets = [v[0] for v in take('<I', n+1)]
ids = [v[0] for v in take('<I', total)]
positions = take('<4f', n)
state = Path('reports/chrono-adami-native.bin').read_bytes()
rho = list(struct.iter_unpack('<4f', state[:n*16]))
assert all(v[0] == 0 for v in struct.iter_unpack('<f', state[n*16:n*28]))
p = {k.replace('constant.paramsD.', ''):v for k,v in json.loads(Path('reports/chrono-params.json').read_text()).items()}
assert p['kernel_type'] == 1 and p['use_delta_sph'] and p['viscosity_method'] == 1
components = json.loads(Path('reports/chrono-rhs-isolated-gpu.json').read_text())['sections'][0]['components']
results = []
for observed in components:
    i, component = divmod(observed['index'], 4)
    value, scale = [0.0]*4, [0.0]*4
    for j in ids[offsets[i]:offsets[i+1]]:
        if i == j: continue
        d = [positions[i][a]-positions[j][a] for a in range(3)]
        for a, axis in enumerate('xyz'):
            if p[axis+'_periodic']:
                period = p['boxDims.'+axis]
                d[a] -= period*round(d[a]/period)
        if sum(x*x for x in d) < (p['epsMinMarkersDis']*p['h'])**2:
            d = [p['epsMinMarkersDis']*p['h'], 0, 0]
        dd = sum(x*x for x in d)
        if dd > (p['h_multiplier']*p['h'])**2: continue
        q = math.sqrt(dd)*p['ooh']
        beta = 3*.31830988618379*p['ooh']**5/4
        factor = beta*(3*q-4) if q<1 else beta*(4-q-4/q) if q<2 else 0
        grad = [factor*x for x in d]
        pressure = -p['markerMass']*(rho[i][1]/rho[i][0]**2+rho[j][1]/rho[j][0]**2)
        terms = [pressure*g for g in grad]
        psi = p['density_delta']*p['h']*p['Cs']*p['markerMass']/rho[j][0]*2*(rho[i][0]-rho[j][0])/(dd+p['epsMinMarkersDis']*p['h']**2)
        terms.append(psi*sum(x*g for x,g in zip(d,grad)))
        for a, term in enumerate(terms):
            value[a] += term
            scale[a] += abs(term)
    for a, axis in enumerate('xyz'):
        value[a] += p['gravity.'+axis]+p['bodyForce3.'+axis]
    results.append(dict(marker=i, component=component, doubleReference=value[component], native=observed['expected'], webgpu=observed['actual'], sumAbsoluteContributions=scale[component]))
Path('reports/chrono-rhs-roundoff.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
