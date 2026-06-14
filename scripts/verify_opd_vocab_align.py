#!/usr/bin/env python3
"""Standalone tokenizer / vocab alignment check for cross-model OPD."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor

from opsd_utils.vocab_align import print_vocab_align_report, verify_shared_tokenizer_alignment


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify student/teacher shared vocab alignment")
    parser.add_argument("--student", required=True, help="Student model id or path")
    parser.add_argument("--teacher", required=True, help="Teacher model id or path")
    parser.add_argument("--full-scan", action="store_true", help="Check every token id (slow)")
    parser.add_argument("--stride", type=int, default=500, help="Sample stride when not full-scan")
    args = parser.parse_args()

    student_proc = AutoProcessor.from_pretrained(args.student)
    teacher_proc = AutoProcessor.from_pretrained(args.teacher)
    shared = min(len(student_proc.tokenizer), len(teacher_proc.tokenizer))
    report = verify_shared_tokenizer_alignment(
        student_proc.tokenizer,
        teacher_proc.tokenizer,
        shared_vocab=shared,
        full_scan=args.full_scan,
        sample_stride=args.stride,
    )
    print_vocab_align_report(report)
    return 0 if report["aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
