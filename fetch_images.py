import os, io, time, random
import requests
from PIL import Image
from tqdm import tqdm

ROOT = "data"
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
SPLITS = {"train": 32, "val": 8}  # Anzahl Bilder pro Klasse und Split
os.makedirs(ROOT, exist_ok=True)
for split in SPLITS:
    for c in CLASSES:
        os.makedirs(os.path.join(ROOT, split, c), exist_ok=True)

# Einfache, frei nutzbare Beispielbilder-Listen (Mix aus freien Quellen; reine Demo)
URLS = {
    "fist": [
        "https://images.pexels.com/photos/2258244/pexels-photo-2258244.jpeg",
        "https://images.pexels.com/photos/2258250/pexels-photo-2258250.jpeg",
        "https://images.pexels.com/photos/163431/fist-blow-power-wrestling-163431.jpeg",
        "https://images.pexels.com/photos/635356/pexels-photo-635356.jpeg",
        "https://images.pexels.com/photos/2049351/pexels-photo-2049351.jpeg",
    ],
    "hand_open": [
        "https://images.pexels.com/photos/6823412/pexels-photo-6823412.jpeg",
        "https://images.pexels.com/photos/8383448/pexels-photo-8383448.jpeg",
        "https://images.pexels.com/photos/4631077/pexels-photo-4631077.jpeg",
        "https://images.pexels.com/photos/6975063/pexels-photo-6975063.jpeg",
        "https://images.pexels.com/photos/906090/pexels-photo-906090.jpeg",
    ],
    "peace": [
        "https://images.pexels.com/photos/25478093/pexels-photo-25478093.jpeg",
        "hhttps://images.pexels.com/photos/4629624/pexels-photo-4629624.jpeg",
        "https://images.pexels.com/photos/9017564/pexels-photo-9017564.jpeg",
        "https://images.pexels.com/photos/249613/pexels-photo-249613.jpeg",
        "https://images.pexels.com/photos/344738/pexels-photo-344738.jpeg",
    ],
    "thumbs_up": [
        "https://images.pexels.com/photos/327533/pexels-photo-327533.jpeg",
        "https://images.pexels.com/photos/193821/pexels-photo-193821.jpeg",
        "https://images.pexels.com/photos/653429/pexels-photo-653429.jpeg",
        "https://images.pexels.com/photos/3201694/pexels-photo-3201694.jpeg",
        "https://images.pexels.com/photos/7298466/pexels-photo-7298466.jpeg",
    ],
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_image(url, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img
    except Exception as e:
        print("Couldn't load img")
        return None

def save_resized_variants(img, base_path, base_name, count_needed, min_size=(256, 256)):
    saved = 0
    w, h = img.size
    if w < min_size[0] or h < min_size[1]:
        img = img.resize(min_size, Image.BILINEAR)
    for i in range(count_needed):
        # einfache zufällige Crops/Rotationen für Diversität
        out = img.copy()
        # random crop
        if random.random() < 0.6:
            cw = int(w * random.uniform(0.7, 0.95))
            ch = int(h * random.uniform(0.7, 0.95))
            if cw < w and ch < h:
                x0 = random.randint(0, w - cw)
                y0 = random.randint(0, h - ch)
                out = out.crop((x0, y0, x0 + cw, y0 + ch)).resize((256, 256), Image.BILINEAR)
        else:
            out = out.resize((256, 256), Image.BILINEAR)
        # random flip
        if random.random() < 0.5:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        # slight rotation
        if random.random() < 0.3:
            ang = random.uniform(-10, 10)
            out = out.rotate(ang, resample=Image.BILINEAR, expand=False)
            out = out.resize((256, 256), Image.BILINEAR)
        # random color jitter
        if random.random() < 0.3:
            # einfacher Helligkeits-Shift
            factor = random.uniform(0.85, 1.15)
            out = Image.eval(out, lambda px: int(max(0, min(255, px * factor))))
        out.save(f"{base_path}_{base_name}_{i:03d}.jpg", quality=90)
        saved += 1
    return saved

def fill_split(split, per_class):
    for cls in CLASSES:
        dest_dir = os.path.join(ROOT, split, cls)
        existing = len([f for f in os.listdir(dest_dir) if f.lower().endswith(".jpg")])
        need = per_class - existing
        if need <= 0:
            continue
        urls = URLS[cls]
        per_url = max(1, need // len(urls))
        pbar = tqdm(urls, desc=f"{split}/{cls}", ncols=80)
        for idx, url in enumerate(pbar):
            img = fetch_image(url)
            if img is None:
                continue
            base_path = os.path.join(dest_dir, f"{split}")
            save_resized_variants(img, base_path, f"{idx}", per_url)
        # falls noch Bedarf, verteile rest
        remaining = per_class - len([f for f in os.listdir(dest_dir) if f.lower().endswith(".jpg")])
        k = 0
        while remaining > 0 and k < len(urls)*3:
            url = random.choice(urls)
            img = fetch_image(url)
            if img:
                base_path = os.path.join(dest_dir, f"{split}_extra")
                save_resized_variants(img, base_path, f"{k}", 1)
                remaining -= 1
            k += 1

def main():
    for split, n in SPLITS.items():
        fill_split(split, n)
    print("Fertig. Beispielbilder in data/train|val pro Klasse gelegt.")

if __name__ == "__main__":
    main()
 