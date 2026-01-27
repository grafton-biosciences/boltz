#!/usr/bin/env python3
"""
Convert the specific target_0.csv file to A3M format
"""

import csv
import os


def convert_target_csv():
    """Convert target_0.csv to A3M format"""

    input_file = "target_0.csv"
    output_file = "target_0.a3m"

    print(f"🔄 Converting {input_file}")
    print(f"   Output: {output_file}")

    sequence_count = 0

    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            reader = csv.DictReader(infile)

            for row_num, row in enumerate(reader):
                # Extract key and sequence from CSV
                key = row.get("key", f"seq_{row_num}")
                sequence = row.get("sequence", "")

                if sequence:
                    # Write A3M format: >header followed by sequence
                    outfile.write(f">{key}\n")
                    outfile.write(f"{sequence}\n")
                    sequence_count += 1

        print(f"✅ Successfully converted {sequence_count} sequences")

        # Validate output
        with open(output_file, "r") as f:
            lines = f.readlines()

        header_count = sum(1 for line in lines if line.startswith(">"))
        sequence_lines = [
            line for line in lines if not line.startswith(">") and line.strip()
        ]

        print(f"\n📊 A3M File Details:")
        print(f"   Headers: {header_count}")
        print(f"   Sequence lines: {len(sequence_lines)}")
        print(f"   Output file: {output_file}")

        if header_count > 0 and len(sequence_lines) > 0:
            print(f"   Sample header: {lines[0].strip()}")
            print(f"   Sample sequence: {sequence_lines[0][:60].strip()}...")
            print(f"✅ Conversion successful!")

        return True

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


if __name__ == "__main__":
    convert_target_csv()
