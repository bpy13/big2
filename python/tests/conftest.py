import sys
from pathlib import Path

# Make `tests.reference` importable as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
