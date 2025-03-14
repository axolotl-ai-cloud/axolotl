#!/usr/bin/env python3
"""
Script to generate API documentation for Axolotl using quartodoc.
"""

import os
import subprocess  # nosec B404
import sys


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=False)  # nosec B602
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(result.returncode)
    return result


def main():
    """Generate API documentation for Axolotl."""
    # Ensure we're in the project root
    if not os.path.exists("_quarto.yml"):
        print("Error: _quarto.yml not found. Run this script from the project root.")
        sys.exit(1)

    # Create the output directories if they don't exist
    os.makedirs("docs/api", exist_ok=True)

    # Generate the documentation
    print("Generating API documentation...")
    run_command("quartodoc build")

    print("Documentation generated successfully!")
    print("Run 'quarto preview' to view the documentation.")


if __name__ == "__main__":
    main()
