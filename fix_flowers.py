"""Rename flowers_102 numeric class folders to canonical flower species names.

The HF mirror stores class labels as numeric strings ("1", "29", ...) which
sort alphabetically into garbage. The standard Oxford Flowers 102 mapping
gives the species name for each numeric id 1..102.
"""

import os
import shutil
from pathlib import Path

# Canonical Oxford Flowers 102 cat_to_name (1-indexed in the original release).
CAT_TO_NAME = {
    "1": "pink primrose", "2": "hard-leaved pocket orchid", "3": "canterbury bells",
    "4": "sweet pea", "5": "english marigold", "6": "tiger lily", "7": "moon orchid",
    "8": "bird of paradise", "9": "monkshood", "10": "globe thistle", "11": "snapdragon",
    "12": "colt's foot", "13": "king protea", "14": "spear thistle", "15": "yellow iris",
    "16": "globe-flower", "17": "purple coneflower", "18": "peruvian lily",
    "19": "balloon flower", "20": "giant white arum lily", "21": "fire lily",
    "22": "pincushion flower", "23": "fritillary", "24": "red ginger", "25": "grape hyacinth",
    "26": "corn poppy", "27": "prince of wales feathers", "28": "stemless gentian",
    "29": "artichoke", "30": "sweet william", "31": "carnation", "32": "garden phlox",
    "33": "love in the mist", "34": "mexican aster", "35": "alpine sea holly",
    "36": "ruby-lipped cattleya", "37": "cape flower", "38": "great masterwort",
    "39": "siam tulip", "40": "lenten rose", "41": "barbeton daisy", "42": "daffodil",
    "43": "sword lily", "44": "poinsettia", "45": "bolero deep blue", "46": "wallflower",
    "47": "marigold", "48": "buttercup", "49": "oxeye daisy", "50": "common dandelion",
    "51": "petunia", "52": "wild pansy", "53": "primula", "54": "sunflower",
    "55": "pelargonium", "56": "bishop of llandaff", "57": "gaura", "58": "geranium",
    "59": "orange dahlia", "60": "pink-yellow dahlia", "61": "cautleya spicata",
    "62": "japanese anemone", "63": "black-eyed susan", "64": "silverbush",
    "65": "californian poppy", "66": "osteospermum", "67": "spring crocus",
    "68": "bearded iris", "69": "windflower", "70": "tree poppy", "71": "gazania",
    "72": "azalea", "73": "water lily", "74": "rose", "75": "thorn apple",
    "76": "morning glory", "77": "passion flower", "78": "lotus lotus", "79": "toad lily",
    "80": "anthurium", "81": "frangipani", "82": "clematis", "83": "hibiscus",
    "84": "columbine", "85": "desert-rose", "86": "tree mallow", "87": "magnolia",
    "88": "cyclamen", "89": "watercress", "90": "canna lily", "91": "hippeastrum",
    "92": "bee balm", "93": "ball moss", "94": "foxglove", "95": "bougainvillea",
    "96": "camellia", "97": "mallow", "98": "mexican petunia", "99": "bromelia",
    "100": "blanket flower", "101": "trumpet creeper", "102": "blackberry lily",
}


def main():
    root = Path("flowers_102/images")
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    existing = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"Found {len(existing)} class folders: {existing[:5]}... {existing[-5:]}")

    # the HF mirror may use 0-indexed (0..101) or 1-indexed (1..102) — handle both
    sample = existing[0]
    try:
        sample_int = int(sample)
    except ValueError:
        raise SystemExit(f"Unexpected folder name: {sample!r}. Already renamed?")
    offset = 1 if sample_int == 0 else 0  # if folders start at 0, shift by 1 to match cat_to_name

    renamed = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        try:
            idx = int(d.name) + offset
        except ValueError:
            continue
        name = CAT_TO_NAME.get(str(idx))
        if not name:
            print(f"  no mapping for index {idx} (folder {d.name})")
            continue
        new = root / name.replace(" ", "_")
        if new.exists():
            print(f"  target exists, skipping: {new}")
            continue
        shutil.move(str(d), str(new))
        renamed += 1
    print(f"Renamed {renamed} folders to canonical species names.")


if __name__ == "__main__":
    main()
