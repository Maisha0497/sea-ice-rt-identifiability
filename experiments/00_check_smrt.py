from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forward_smrt import smrt_metadata


def main():
    metadata = smrt_metadata()
    print("SMRT import succeeded.")
    print(f"SMRT version: {metadata['version']}")
    print(f"SMRT imported from: {metadata['module_file']}")


if __name__ == "__main__":
    main()
