#!/usr/bin/env python3
"""
Checkpoint Comparison Tool

Compares checkpoint files from two directories (e.g., CPU vs GPU implementations)
to verify that both implementations produce identical simulation results.

This script:
1. Checks that both directories contain the same number of checkpoint files
2. Verifies that checkpoint filenames match between directories
3. Compares the content of corresponding checkpoint files (STEP, CHECKSUM, ALIVE_CELLS)
4. Reports any discrepancies found

Usage:
    python compare_checkpoints.py <dir1> <dir2>
    python compare_checkpoints.py checkpoints/cpu checkpoints/gpu

Exit codes:
    0 - All checkpoints match
    1 - Checkpoints differ or comparison failed
"""

import argparse
import os
import sys
from pathlib import Path


def parse_checkpoint_file(filepath):
    """Parse a checkpoint file and return its contents as a dictionary."""
    data = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def get_checkpoint_files(directory):
    """Get sorted list of checkpoint files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        return None

    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a directory", file=sys.stderr)
        return None

    # Get all .dat files and sort them
    checkpoint_files = sorted([f.name for f in dir_path.glob('*.dat')])
    return checkpoint_files


def compare_directories(dir1, dir2, verbose=False):
    """Compare checkpoint files in two directories."""
    print(f"Comparing checkpoints:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    print()

    # Get checkpoint files from both directories
    files1 = get_checkpoint_files(dir1)
    files2 = get_checkpoint_files(dir2)

    if files1 is None or files2 is None:
        return False

    # Check if both directories have files
    if not files1 and not files2:
        print("Warning: Both directories are empty")
        return True

    if not files1:
        print(f"Error: No checkpoint files found in {dir1}", file=sys.stderr)
        return False

    if not files2:
        print(f"Error: No checkpoint files found in {dir2}", file=sys.stderr)
        return False

    # Check if the number of files matches
    if len(files1) != len(files2):
        print(f"Error: Number of checkpoint files differs!")
        print(f"  {dir1}: {len(files1)} files")
        print(f"  {dir2}: {len(files2)} files")
        print()
        print(f"Files in {dir1} only: {set(files1) - set(files2)}")
        print(f"Files in {dir2} only: {set(files2) - set(files1)}")
        return False

    print(f"Found {len(files1)} checkpoint files in each directory")
    print()

    # Check if filenames match
    if set(files1) != set(files2):
        print("Error: Checkpoint filenames do not match!")
        print(f"Files in {dir1} only: {set(files1) - set(files2)}")
        print(f"Files in {dir2} only: {set(files2) - set(files1)}")
        return False

    # Compare contents of each file pair
    all_match = True
    mismatches = []

    for filename in files1:
        file1_path = Path(dir1) / filename
        file2_path = Path(dir2) / filename

        data1 = parse_checkpoint_file(file1_path)
        data2 = parse_checkpoint_file(file2_path)

        if data1 is None or data2 is None:
            all_match = False
            continue

        # Compare the contents
        if data1 != data2:
            all_match = False
            mismatches.append({
                'filename': filename,
                'data1': data1,
                'data2': data2
            })

            if verbose:
                print(f"✗ {filename}: MISMATCH")
                print(f"  {dir1}:")
                for key, value in data1.items():
                    print(f"    {key}: {value}")
                print(f"  {dir2}:")
                for key, value in data2.items():
                    print(f"    {key}: {value}")
                print()
        else:
            if verbose:
                print(f"✓ {filename}: MATCH")

    # Print summary
    print()
    print("=" * 60)
    if all_match:
        print("SUCCESS: All checkpoint files match!")
        print(f"Verified {len(files1)} checkpoint files")
        return True
    else:
        print("FAILURE: Checkpoints differ!")
        print(f"Mismatches found in {len(mismatches)} file(s):")
        for mismatch in mismatches:
            print(f"  - {mismatch['filename']}")
            # Show what differs
            keys1 = set(mismatch['data1'].keys())
            keys2 = set(mismatch['data2'].keys())
            all_keys = keys1 | keys2

            for key in all_keys:
                val1 = mismatch['data1'].get(key, 'MISSING')
                val2 = mismatch['data2'].get(key, 'MISSING')
                if val1 != val2:
                    print(f"    {key}: {val1} != {val2}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare checkpoint files from two directories to verify simulation correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoints/cpu checkpoints/gpu
  %(prog)s --verbose cpu_output gpu_output

This tool is used to verify that CPU and GPU implementations of the Game of Life
produce identical results by comparing their checkpoint files.
        """
    )

    parser.add_argument('dir1', help='First checkpoint directory')
    parser.add_argument('dir2', help='Second checkpoint directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed comparison for each file')

    args = parser.parse_args()

    # Compare the directories
    success = compare_directories(args.dir1, args.dir2, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
