"""CPU check that every command-line script imports and parses its arguments (``--help``).

    python tests/test_scripts.py
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    scripts = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))
    assert len(scripts) >= 12, scripts
    for s in scripts:
        r = subprocess.run([sys.executable, s, "--help"], capture_output=True, text=True)
        assert r.returncode == 0 and "usage:" in r.stdout, f"{s} --help failed:\n{r.stderr[-2000:]}"
        print(f"  {os.path.relpath(s, ROOT)} --help ok")
    print(f"scripts: {len(scripts)} entry points import and parse")


if __name__ == "__main__":
    main()
