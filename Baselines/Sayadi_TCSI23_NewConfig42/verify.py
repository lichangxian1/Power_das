#!/usr/bin/env python3
"""RTL == golden equivalence check (verilator).

1) sanity: comp_verilog exprs == comp_eval truth tables (pure python)
2) golden vectors (2M random + corners) -> vec files
3) verilate rtl/sayadi_mul{1,2}_16.v with tb_check.cpp, run, expect PASS
"""
import itertools
import json
import os
import subprocess
import numpy as np
from sayadi_common import build_schedule, simulate, comp_eval, comp_verilog

HERE = os.path.dirname(os.path.abspath(__file__))


def check_exprs():
    for fam, ks in (('AC6G', range(1, 17)), ('ACFGI', range(1, 5)),
                    ('ACFGII', range(1, 13)), ('FA', [0]), ('HA', [0])):
        for k in ks:
            for bits in itertools.product((0, 1), repeat=4):
                s_ref, c_ref = comp_eval(fam, k, bits)
                sv, cv = comp_verilog(fam, k, [str(x) for x in bits])
                env = {}
                s_got = eval(sv.replace("1'b1", "1").replace("1'b0", "0"), env)
                c_got = eval(cv.replace("1'b1", "1").replace("1'b0", "0"), env)
                assert (int(s_ref) & 1, int(c_ref) & 1) == (s_got & 1, c_got & 1), \
                    (fam, k, bits, (s_ref, c_ref), (s_got, c_got))
    print("comp_verilog == comp_eval for all compressors: OK")


ASG_FILE = os.environ.get('SAYADI_ASG', 'assignment16.json')


def golden_vectors(mul, fam, n=2_000_000, seed=999):
    with open(os.path.join(HERE, ASG_FILE)) as f:
        raw = json.load(f)[mul]
    asg = {tuple(int(x) for x in k.split(',')): tuple(v) for k, v in raw.items()}
    boxes, final_cols, _ = build_schedule(16, fam)
    rng = np.random.default_rng(seed)
    M = (1 << 16) - 1
    corners = [0, 1, 2, 3, 255, 256, 32767, 32768, M - 1, M]
    ca, cb = zip(*itertools.product(corners, corners))
    a = np.concatenate([np.array(ca, dtype=np.uint32),
                        rng.integers(0, 1 << 16, n, dtype=np.uint32)])
    b = np.concatenate([np.array(cb, dtype=np.uint32),
                        rng.integers(0, 1 << 16, n, dtype=np.uint32)])
    g = simulate(16, boxes, final_cols, asg, a, b).astype(np.int64) & 0x7FFFFFFF
    path = os.path.join(HERE, f'vec_{mul}_16.txt')
    np.savetxt(path, np.stack([a, b, g], axis=1), fmt='%d')
    return path


def run_one(mul, vec):
    rtl = os.path.join(HERE, 'rtl', f'sayadi_{mul}_16.v')
    obj = os.path.join(HERE, f'obj_{mul}')
    subprocess.run(['rm', '-rf', obj], check=True)
    r = subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '4', '-O3',
                        '-Wno-fatal', '--top-module', 'MUL', '--Mdir', obj,
                        rtl, os.path.join(HERE, 'tb_check.cpp'), '-o', 'sim'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([os.path.join(obj, 'sim'), vec], capture_output=True, text=True)
    print(r.stdout.strip())
    assert r.returncode == 0, f"{mul} FAILED"


if __name__ == '__main__':
    check_exprs()
    for mul, fam in (('mul1', 'ACFGI'), ('mul2', 'ACFGII')):
        vec = golden_vectors(mul, fam)
        run_one(mul, vec)
    print("ALL PASS")
