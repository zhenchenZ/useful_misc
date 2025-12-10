import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import Optional

def load_and_tile_imgs(img_dir: Path) -> Image.Image:
    """
    Load all PNGs in a directory and tile them horizontally.
    Used for conditional views (multi-view renders).
    """
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No PNGs found in {img_dir}")

    img_list = []
    for p in img_paths:
        img = Image.open(p)
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])  # alpha as mask
            img_list.append(bg)
        else:
            img_list.append(img.convert('RGB'))

    # init a blank canvas
    total_width = sum(img.width for img in img_list)
    height = img_list[0].height
    tiled_img = Image.new('RGB', (total_width, height))

    x_offset = 0
    for img in img_list:
        tiled_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return tiled_img


def find_common_shas(experiment_dirs: dict[str, Path]) -> list[str]:
    """
    Get the intersection of PNG stems across all experiment dirs.
    """
    sha_sets = []
    for name, d in experiment_dirs.items():
        stems = {p.stem for p in d.glob("*.png")}
        if not stems:
            raise FileNotFoundError(f"No PNGs found in experiment dir {name}: {d}")
        sha_sets.append(stems)

    common = set.intersection(*sha_sets)
    return sorted(common)


def annotate_row(img: Image.Image, text: str, y_offset: int, padding: int = 4) -> None:
    """
    Draw a small label on the image at vertical position y_offset.
    Uses textbbox instead of deprecated textsize.
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text = str(text)
    
    # works in all new Pillow versions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # small white box behind text for readability
    box_x0, box_y0 = 5, y_offset + 5
    box_x1, box_y1 = box_x0 + text_w + 2 * padding, box_y0 + text_h + 2 * padding
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(255, 255, 255))

    # text position
    text_x = box_x0 + padding
    text_y = box_y0 + padding
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)


def concatenate_mesh_renders(
    input_render_dir: Optional[Path],
    experiment_dirs: dict[str, Path],
    output_dir: Path,
    annotate: bool = True,
):
    """
    input_render_dir: root containing conditional-view folders, one subdir per mesh SHA.
                      e.g. input_render_dir / <sha> / view*.png
    experiment_dirs:  dict mapping experiment name -> dir with <sha>.png renders
                      e.g. {"single": Path(".../nv_renders_single_gen"), "multi": Path(".../nv_renders_multi_gen")}
    output_dir:       where concatenated images will be saved
    annotate:         if True, writes experiment names as labels at the start of each row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # common SHAs across all experiments
    sha_list = find_common_shas(experiment_dirs)
    print(f"Found {len(sha_list)} common meshes between the experiment render directories.")

    exp_items = list(experiment_dirs.items())  # list of (exp_name, dir)

    for sha in tqdm(sha_list):
        # 1) conditional views (first row)
        if input_render_dir is not None:
            cond_dir = input_render_dir / sha
            cond_views = load_and_tile_imgs(cond_dir)  # horizontal strip
            row_images = [("cond", cond_views)]  # keep a name even if we don't annotate this row
        else:
            row_images = []

        # 2) experiment rows (one row per experiment)
        for exp_name, exp_dir in exp_items:
            img_path = exp_dir / f"{sha}.png"
            if not img_path.exists():
                # skip if missing in this experiment
                print(f"[WARN] Missing {img_path}, skipping sha {sha} for experiment {exp_name}")
                break
            img = Image.open(img_path).convert('RGB')
            row_images.append((exp_name, img))
        else:
            # only executed if the loop didn't break ⇒ we have all rows
            # compute final canvas size
            widths = [im.width for _, im in row_images]
            heights = [im.height for _, im in row_images]

            combined_width = max(widths)
            combined_height = sum(heights)

            combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

            # paste rows sequentially
            y_offset = 0
            for row_name, row_img in row_images:
                # center row horizontally if narrower than max width
                x_offset = (combined_width - row_img.width) // 2
                combined_img.paste(row_img, (x_offset, y_offset))

                # optional annotation (skip cond row if you prefer)
                if annotate and row_name != "cond":
                    annotate_row(combined_img, row_name, y_offset)

                y_offset += row_img.height

            combined_img.save(output_dir / f"{sha}.png")


# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    """
    Example usage:
    python concat_run_renders.py \
        --root_dir data/100_gtMeshes_insMax50_pickedCls0-1 \
        --output_dir data/100_gtMeshes_insMax50_pickedCls0-1/nv_gen_seg_renders/pipeline08-09-comparison

    """
    parser = argparse.ArgumentParser(description="Concatenate and compare renders from multiple experiments.")
    parser.add_argument("--root_dir", type=str, help="Root directory containing input renders and experiment subdirs.")
    parser.add_argument("--output_dir", type=str, help="Directory to save concatenated comparison renders.")
    parser.add_argument("--input_render_name", type=str, default=None, help="Name of the subdirectory in root_dir containing conditional view renders. If not provided, no conditional views will be included.")    
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    input_render_dir = None if args.input_render_name is None else root_dir / args.input_render_name

    experiment_dirs = {
        "pipeline08_only_cls"           : root_dir / "nv_gen_seg_renders/gen_meshes_multidiffu_pipeline08",
        "pipeline09_cls_instance"       : root_dir / "nv_gen_seg_renders/gen_meshes_multidiffu_pipeline09"
    }

    concatenate_mesh_renders(
        input_render_dir=input_render_dir,
        experiment_dirs=experiment_dirs,
        output_dir=Path(args.output_dir),
        annotate=True,  # set False if you don't want labels
    )
