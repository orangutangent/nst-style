# -*- coding: utf-8 -*-
"""Main file for running image stylization."""

from pathlib import Path
from src.stylize import stylize_image
from src.utils import save_image

# Image paths
ASSETS_DIR = Path("assets")
CONTENT_IMAGE = ASSETS_DIR / "content2.jpg"
STYLE_IMAGE = ASSETS_DIR / "style4.webp"
OUTPUT_IMAGE = ASSETS_DIR / "output.jpg"


def main():
    """Main function for image stylization."""  
    # Check file existence
    if not CONTENT_IMAGE.exists():
        print(f"Error: Content image not found at {CONTENT_IMAGE}")
        print("Please add your content image to the assets folder.")
        return

    if not STYLE_IMAGE.exists():
        print(f"Error: Style image not found at {STYLE_IMAGE}")
        print("Please add your style image to the assets folder.")
        return

    # Run stylization
    output = stylize_image(
        content_path=str(CONTENT_IMAGE),
        style_path=str(STYLE_IMAGE),
        output_size=512,
        num_steps=500,
        style_weight=10_000_000,
        content_weight=1
    )

    # Save result
    save_image(output, str(OUTPUT_IMAGE))


if __name__ == "__main__":
    main()

