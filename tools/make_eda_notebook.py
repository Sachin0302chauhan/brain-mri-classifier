"""Generate notebooks/01_data_exploration.ipynb (run once, then execute it)."""

from pathlib import Path

import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell(
        "# Brain MRI dataset: exploratory analysis\n\n"
        "Before training, check what the data looks like: how many images each "
        "class has, how image sizes vary, and what typical scans from each class "
        "look like. These checks decide the preprocessing in `brainmri/data.py`."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n"
        "from brainmri import config\n\n"
        "rows = []\n"
        "for split, folder in [('train', config.TRAIN_DIR), ('test', config.TEST_DIR)]:\n"
        "    for cls in config.CLASSES:\n"
        "        for path in sorted((folder / cls).iterdir()):\n"
        "            with Image.open(path) as img:\n"
        "                rows.append(dict(split=split, cls=config.CLASS_LABELS[cls],\n"
        "                                 path=path, width=img.width, height=img.height,\n"
        "                                 mode=img.mode))\n"
        "df = pd.DataFrame(rows)\n"
        "df.head()"
    ),
    nbf.v4.new_markdown_cell("## Class balance"),
    nbf.v4.new_code_cell(
        "counts = df.pivot_table(index='cls', columns='split', values='path', aggfunc='count')\n"
        "counts.plot.bar(figsize=(7, 4), rot=0, title='Images per class')\n"
        "plt.ylabel('Images'); plt.xlabel(''); plt.show()\n"
        "counts"
    ),
    nbf.v4.new_markdown_cell(
        "The classes are balanced, so plain accuracy is a fair headline metric; "
        "macro F1 is reported alongside it anyway."
    ),
    nbf.v4.new_markdown_cell("## Image sizes and colour modes"),
    nbf.v4.new_code_cell(
        "print(df['mode'].value_counts())\n"
        "print()\n"
        "print(df[['width', 'height']].describe().round(0))\n"
        "df.plot.scatter(x='width', y='height', alpha=0.2, figsize=(5, 5),\n"
        "                title='Image dimensions')\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "Sizes vary widely and some images are stored as RGB while others are "
        "greyscale. The pipeline therefore converts every image to greyscale "
        "(replicated to 3 channels for the ImageNet backbone) and resizes to "
        "224×224."
    ),
    nbf.v4.new_markdown_cell("## Example scans"),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(4, 5, figsize=(12, 10))\n"
        "train = df[df.split == 'train']\n"
        "for row, cls in zip(axes, train.cls.unique()):\n"
        "    for ax, path in zip(row, train[train.cls == cls].sample(5, random_state=0).path):\n"
        "        ax.imshow(Image.open(path).convert('L'), cmap='gray')\n"
        "        ax.set_axis_off()\n"
        "    row[0].set_title(cls, loc='left')\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "- Balanced classes: no resampling or class weights needed.\n"
        "- Mixed sizes and colour modes: normalise to greyscale, 224×224.\n"
        "- Scans come in different planes (axial, coronal, sagittal) and "
        "framings: augment with crops, flips, small rotations and "
        "brightness/contrast changes."
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "01_data_exploration.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Wrote {out}")
