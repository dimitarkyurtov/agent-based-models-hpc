#!/usr/bin/env python3
"""
Checkpoint Comparison Tool

Compares checkpoint files from two simulation runs to verify that both
implementations produce identical results. This tool is agnostic to the
specific agent-based model being simulated.

Checkpoint Format:
Each line in a checkpoint file contains: <step_number> <checksum_hash>

This script:
1. Parses checkpoint files containing step-by-step simulation state hashes
2. Compares corresponding steps between two checkpoint files
3. Reports any discrepancies in step numbers or checksums
4. Verifies that both files contain the same number of checkpoints

Usage:
    python compare_checkpoints.py <file1> <file2>
    python compare_checkpoints.py checkpoints/cpu/checkpoints.dat checkpoints/gpu/checkpoints.dat

Exit codes:
    0 - All checkpoints match
    1 - Checkpoints differ or comparison failed
"""

import argparse
import os
import sys
from pathlib import Path


def parse_checkpoint_file(filepath):
    """Parse a checkpoint file and return a list of (step, checksum) tuples."""
    checkpoints = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Invalid format at {filepath}:{line_num}: '{line}'", file=sys.stderr)
                    continue

                try:
                    step = int(parts[0])
                    checksum = parts[1]
                    checkpoints.append((step, checksum))
                except ValueError:
                    print(f"Warning: Invalid step number at {filepath}:{line_num}: '{parts[0]}'", file=sys.stderr)
                    continue

        return checkpoints
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def compare_checkpoint_files(file1, file2, verbose=False):
    """Compare two checkpoint files."""
    print(f"Comparing checkpoint files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print()

    file1_path = Path(file1)
    file2_path = Path(file2)

    if not file1_path.exists():
        print(f"Error: File '{file1}' does not exist", file=sys.stderr)
        return False

    if not file2_path.exists():
        print(f"Error: File '{file2}' does not exist", file=sys.stderr)
        return False

    checkpoints1 = parse_checkpoint_file(file1_path)
    checkpoints2 = parse_checkpoint_file(file2_path)

    if checkpoints1 is None or checkpoints2 is None:
        return False

    if not checkpoints1 and not checkpoints2:
        print("Warning: Both files are empty")
        return True

    if not checkpoints1:
        print(f"Error: No valid checkpoints found in {file1}", file=sys.stderr)
        return False

    if not checkpoints2:
        print(f"Error: No valid checkpoints found in {file2}", file=sys.stderr)
        return False

    # Check if the number of checkpoints matches
    if len(checkpoints1) != len(checkpoints2):
        print(f"Error: Number of checkpoints differs!")
        print(f"  {file1}: {len(checkpoints1)} checkpoints")
        print(f"  {file2}: {len(checkpoints2)} checkpoints")
        return False

    print(f"Found {len(checkpoints1)} checkpoints in each file")
    print()

    all_match = True
    mismatches = []

    for i, ((step1, hash1), (step2, hash2)) in enumerate(zip(checkpoints1, checkpoints2)):
        if step1 != step2:
            all_match = False
            mismatches.append({
                'index': i,
                'issue': 'step_mismatch',
                'step1': step1,
                'step2': step2,
                'hash1': hash1,
                'hash2': hash2
            })
            if verbose:
                print(f"✗ Checkpoint {i}: STEP MISMATCH")
                print(f"  {file1}: step={step1}, hash={hash1}")
                print(f"  {file2}: step={step2}, hash={hash2}")
                print()
        elif hash1 != hash2:
            all_match = False
            mismatches.append({
                'index': i,
                'issue': 'hash_mismatch',
                'step': step1,
                'hash1': hash1,
                'hash2': hash2
            })
            if verbose:
                print(f"✗ Checkpoint {i} (step {step1}): HASH MISMATCH")
                print(f"  {file1}: {hash1}")
                print(f"  {file2}: {hash2}")
                print()
        else:
            if verbose:
                print(f"✓ Checkpoint {i} (step {step1}): MATCH")

    # Print summary
    print()
    print("=" * 60)
    if all_match:
        print("SUCCESS: All checkpoints match!")
        print(f"Verified {len(checkpoints1)} checkpoints")
        return True
    else:
        print("FAILURE: Checkpoints differ!")
        print(f"Mismatches found in {len(mismatches)} checkpoint(s):")
        for mismatch in mismatches:
            if mismatch['issue'] == 'step_mismatch':
                print(f"  - Index {mismatch['index']}: Step mismatch ({mismatch['step1']} != {mismatch['step2']})")
            else:
                print(f"  - Index {mismatch['index']} (step {mismatch['step']}): Hash mismatch")
                print(f"      {file1}: {mismatch['hash1']}")
                print(f"      {file2}: {mismatch['hash2']}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare checkpoint files from two simulation runs to verify correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoints/cpu/checkpoints.dat checkpoints/gpu/checkpoints.dat
  %(prog)s --verbose run1/output.dat run2/output.dat

This tool verifies that two simulation runs produce identical results by comparing
their checkpoint files. It is agnostic to the specific agent-based model being simulated.
        """
    )

    parser.add_argument('file1', help='First checkpoint file')
    parser.add_argument('file2', help='Second checkpoint file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed comparison for each checkpoint')

    args = parser.parse_args()

    # Compare the checkpoint files
    success = compare_checkpoint_files(args.file1, args.file2, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
