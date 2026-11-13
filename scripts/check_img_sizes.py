from pathlib import Path
from PIL import Image

for p in sorted(Path("work/charts").glob("*.png")):
    with Image.open(p) as im:
        w, h = im.size
        print(f"{p.name}: {w}x{h} (ratio={w/h:.2f})")
