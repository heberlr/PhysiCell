"""Test whether "save + resume + continue" reproduces a straight-through run.

This exercises the "potential bitwise restart" claim in
protocols/counter_based_rng.md: with counter_based_rng enabled, random_step is
derived from the simulated time t (not a hidden in-memory counter), so a
resumed run's RNG draws should line up with an uninterrupted run's, provided
cell IDs and current_time both restore correctly across the save/resume
boundary.

Requires a project built from unit_tests/resume_sim/main.cpp (the only main.cpp
in this repo wired for resume -- see unit_tests/resume_sim/README.md). The
plain sample-project main.cpp does NOT support resume.

Usage:
    python beta/test_restart_repro.py <executable> <config_file> <t_mid> <t_final> <rng_mode> <work_dir> [threads]

Example:
    python beta/test_restart_repro.py project unit_tests/resume_sim/config/cycle_phase_3cells_custom_vecs.xml 240 480 counter_based local_runs
"""

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyMCDS import pyMCDS


RANDOM_SEED = "17"


def update_config(xml_file, output_dir, threads, max_time, rng_mode):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    root.find(".//overall/max_time").text = str(max_time)
    root.find(".//parallel/omp_num_threads").text = str(threads)

    save_folder = root.find(".//save/folder")
    if save_folder is None:
        save_node = root.find(".//save")
        if save_node is None:
            raise RuntimeError("Could not find <save> in config file")
        save_folder = ET.SubElement(save_node, "folder")
    save_folder.text = str(output_dir)

    options = root.find(".//options")
    if options is None:
        options = ET.SubElement(root, "options")

    random_seed = options.find("random_seed")
    if random_seed is None:
        random_seed = ET.SubElement(options, "random_seed")
    random_seed.text = RANDOM_SEED

    rng_mode_node = options.find("rng_mode")
    if rng_mode is None:
        if rng_mode_node is not None:
            options.remove(rng_mode_node)
    else:
        if rng_mode_node is None:
            rng_mode_node = ET.SubElement(options, "rng_mode")
        rng_mode_node.text = rng_mode

    tree.write(xml_file)


def run_once(repo_root, executable, *args):
    cmd = [str(repo_root / executable)] + [str(a) for a in args]
    print(f"\n running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, check=True)


def diff_svg(file_a, file_b):
    """Same tolerance rule as test_diff_svg.py: allow only the trailing
    '0 days ... seconds' wall-clock-time line to differ."""
    cmd = ["diff", str(file_a), str(file_b)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode > 1:
        raise RuntimeError(f"Error running diff: {res.stderr.decode()}")
    lines = res.stdout.splitlines()
    if len(lines) == 0:
        return True, b""
    if len(lines) == 4 and lines[1].startswith(b"<    0 days") and lines[3].startswith(b">    0 days"):
        return True, b""
    return False, res.stdout


def check_no_duplicate_live_cell_ids(xml_file, output_dir):
    """Best-effort check for the cell-ID-collision gap found in resume:
    max_basic_agent_ID is reset from the *count* of resumed cells, not the
    true historical max ID ever issued, so if any cells died/were removed
    before the checkpoint, IDs assigned to new cells after resume can collide
    with IDs already in use by cells that were alive at resume time. This
    only actually triggers if the scenario had deaths/removals before the
    checkpoint -- a clean pass here does not by itself rule out the gap for
    scenarios without that condition.
    """
    mcds = pyMCDS(xml_file, str(output_dir))
    df = mcds.get_cell_df()

    if "ID" not in df.columns:
        print(f"  (skipping ID-collision check: no 'ID' column in {xml_file})")
        return True

    if "dead" in df.columns:
        alive = df[df["dead"] == 0.0]
    else:
        alive = df

    ids = alive["ID"]
    dup_mask = ids.duplicated(keep=False)
    if dup_mask.any():
        print("  DUPLICATE live cell IDs found after resume:")
        print(alive.loc[dup_mask, "ID"].to_string())
        return False

    print(f"  no duplicate live cell IDs among {len(alive)} alive cells")
    return True


def next_output_index(output_dir):
    existing = sorted(output_dir.glob("output????????.xml"))
    if not existing:
        return 1
    last = existing[-1].stem  # e.g. "output00000003"
    return int(last[len("output"):]) + 1


def compare_post_resume_svgs(straight_output, split_output, next_svg_idx):
    """Diff every snapshot SVG the resumed run wrote (index >= next_svg_idx,
    i.e. written during Run B part 2) against the matching-index snapshot
    from the straight-through run, plus the final.svg pair. Returns True iff
    every pair matched (within the wall-clock-time tolerance of diff_svg)."""
    split_snapshots = sorted(split_output.glob("snapshot????????.svg"))
    post_resume = [
        f for f in split_snapshots
        if int(f.stem[len("snapshot"):]) >= next_svg_idx
    ]

    pairs = [(straight_output / f.name, f) for f in post_resume]
    pairs.append((straight_output / "final.svg", split_output / "final.svg"))

    all_ok = True
    for straight_svg, split_svg in pairs:
        if not straight_svg.exists() or not split_svg.exists():
            print(f" {straight_svg} vs {split_svg}: MISSING")
            all_ok = False
            continue

        ok, diff_output = diff_svg(straight_svg, split_svg)
        if ok:
            print(f" {straight_svg} vs {split_svg}: OK")
        else:
            print(f" {straight_svg} vs {split_svg}: ERR")
            print(diff_output.decode(errors="replace"))
            all_ok = False

    return all_ok


def main(executable, config_file, t_mid, t_final, rng_mode="counter_based", work_dir=None, threads=1):
    repo_root = Path.cwd()
    config_source = repo_root / config_file
    t_mid = float(t_mid)
    t_final = float(t_final)
    threads = int(threads)
    rng_mode_arg = rng_mode.strip().lower() if isinstance(rng_mode, str) else rng_mode
    if rng_mode_arg in ("", "off", "none", "omit", "omitted", "default"):
        rng_mode = None
    if not (0 < t_mid < t_final):
        raise SystemExit("Require 0 < t_mid < t_final")

    if work_dir is None:
        tmp_context = tempfile.TemporaryDirectory(prefix="physicell-restart-repro-")
        tmp_root = Path(tmp_context.__enter__())
        cleanup_context = tmp_context
    else:
        work_root = Path(work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        tmp_root = work_root / f"physicell-restart-repro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        cleanup_context = None

    try:
        # --- Run A: straight through, 0 -> t_final, one invocation ---
        straight_dir = tmp_root / "straight"
        straight_output = straight_dir / "output"
        straight_output.mkdir(parents=True, exist_ok=True)
        straight_config = straight_dir / "PhysiCell_settings.xml"
        shutil.copy(config_source, straight_config)
        update_config(straight_config, straight_output, threads, t_final, rng_mode)

        print(f"\n{'=' * 80}\n Run A (straight through, 0 -> {t_final})")
        run_once(repo_root, executable, straight_config)

        # --- Run B, part 1: 0 -> t_mid, produces the checkpoint to resume from ---
        split_dir = tmp_root / "split"
        split_output = split_dir / "output"
        split_output.mkdir(parents=True, exist_ok=True)
        split_config = split_dir / "PhysiCell_settings.xml"
        shutil.copy(config_source, split_config)
        update_config(split_config, split_output, threads, t_mid, rng_mode)

        print(f"\n{'=' * 80}\n Run B part 1 (0 -> {t_mid}, checkpoint)")
        run_once(repo_root, executable, split_config)

        checkpoint_xml = split_output / "final.xml"
        if not checkpoint_xml.exists():
            raise SystemExit(f"Expected checkpoint not found: {checkpoint_xml}")
        checkpoint_time = pyMCDS("final.xml", str(split_output)).get_time()
        print(f" checkpoint current_time = {checkpoint_time} (target t_mid = {t_mid})")

        # --- Run B, part 2: resume from the checkpoint, continue to t_final ---
        next_idx = next_output_index(split_output)
        update_config(split_config, split_output, threads, t_final, rng_mode)

        resume_folder = split_output.relative_to(repo_root) if split_output.is_relative_to(repo_root) else split_output

        print(f"\n{'=' * 80}\n Run B part 2 (resume from final.xml, {t_mid} -> {t_final})")
        run_once(repo_root, executable, split_config, resume_folder, "final.xml", next_idx, next_idx)

        # --- Compare final state ---
        straight_final_xml = straight_output / "final.xml"
        split_final_xml = split_output / "final.xml"
        straight_final_svg = straight_output / "final.svg"
        split_final_svg = split_output / "final.svg"

        for f in (straight_final_xml, split_final_xml, straight_final_svg, split_final_svg):
            if not f.exists():
                raise SystemExit(f"Expected output file not found: {f}")

        print(f"\n{'=' * 80}\n Comparing every post-resume SVG at t = {t_final}")

        ok_svg = compare_post_resume_svgs(straight_output, split_output, next_idx)

        ok_ids = check_no_duplicate_live_cell_ids("final.xml", split_output)

        if ok_svg and ok_ids:
            print(f"\n{'=' * 80}\n restart reproducibility test passed")
        else:
            raise SystemExit(f"\n{'=' * 80}\n restart reproducibility test FAILED")
    finally:
        if cleanup_context is not None:
            cleanup_context.__exit__(None, None, None)


if __name__ == "__main__":
    main(*sys.argv[1:])
