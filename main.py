# -*- coding: utf-8 -*-
"""Main file for running image stylization."""

import argparse
from pathlib import Path
from src.stylize import stylize_image
from src.utils import save_image


def main():
    """Main function for image stylization with CLI."""
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer - Transfer style from one image to another",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --content content.jpg --style style.jpg
  python main.py --content content.jpg --style style.jpg --output result.jpg --size 1024
  python main.py -c content.jpg -s style.jpg -o result.jpg --steps 500 --style-weight 2000000
        """
    )

    # Required arguments
    parser.add_argument(
        "--content", "-c",
        type=str,
        required=True,
        help="Path to content image"
    )
    parser.add_argument(
        "--style", "-s",
        type=str,
        required=True,
        help="Path to style image"
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.jpg",
        help="Path to save output image (default: output.jpg)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size (default: 512)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of optimization steps (default: 300)"
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1_000_000,
        help="Style loss weight (default: 1000000)"
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1,
        help="Content loss weight (default: 1)"
    )

    args = parser.parse_args()

    # Validate input files
    content_path = Path(args.content)
    style_path = Path(args.style)

    if not content_path.exists():
        print(f"Error: Content image not found at {content_path}")
        return 1

    if not style_path.exists():
        print(f"Error: Style image not found at {style_path}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Content image: {content_path}")
    print(f"Style image: {style_path}")
    print(f"Output image: {output_path}")
    print(f"Size: {args.size}, Steps: {args.steps}")
    print(f"Style weight: {args.style_weight}, Content weight: {args.content_weight}")
    print("-" * 50)

    # Run stylization
    try:
        output = stylize_image(
            content_path=str(content_path),
            style_path=str(style_path),
            output_size=args.size,
            num_steps=args.steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight
        )

        # Save result
        save_image(output, str(output_path))
        print(f"\nSuccess! Stylized image saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"\nError during stylization: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

