import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

scripts = [
    "scripts/fig2A.py",
    "scripts/fig2BandfigS5.py",
    "scripts/fig2C.py",
    "scripts/fig2D.py",
    "scripts/fig3A.py",
    "scripts/fig3B.py",
    "scripts/fig3C.py",
    "scripts/fig3D.py",
    "scripts/fig4AB.py",
    "scripts/fig4C1.py",
    "scripts/fig4C2.py",
    "scripts/fig4C3.py",
    "scripts/fig4C4.py",
    "scripts/figS1.py",
    "scripts/figS2A.py",
    "scripts/figS2B.py",
    "scripts/figS2C.py",
    "scripts/figS2D.py",
    "scripts/figS3A.py",
    "scripts/figS3B.py",
    "scripts/figS3C.py",
    "scripts/figS3D.py",
    "scripts/figS4.py",
    "scripts/figS6.py",
]

for rel in scripts:
    script = ROOT / rel

    if not script.exists():
        raise FileNotFoundError(script)

    print(f"Running {rel}")
    subprocess.run([sys.executable, str(script)], check=True)

print("All figures generated successfully.")