from pathlib import Path

# 仓库根目录
ROOT = Path(__file__).resolve().parent

# 各个子目录
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
FIGURES = ROOT / "figures"

# 如果不存在自动创建
FIGURES.mkdir(exist_ok=True)