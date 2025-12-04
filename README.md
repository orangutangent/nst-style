# Neural Style Transfer Project

Project for transferring style from one image to another using neural networks.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Create Virtual Environment

**Linux/macOS:**

```bash
python3 -m venv venv
```

**Windows:**

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Linux/macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

After activation, you should see `(venv)` prefix in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- PyTorch (with CUDA/MPS support if available)
- torchvision
- Pillow
- matplotlib
- numpy

### Step 5: Verify Installation

You can verify that everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Deactivating Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

## Usage

### Command Line Interface (CLI)

The easiest way to use the project is through the command line interface:

**Basic usage:**

```bash
python main.py --content content.jpg --style style.jpg
```

**With all options:**

```bash
python main.py \
  --content assets/content2.jpg \
  --style assets/style4.webp \
  --output result.jpg \
  --size 512 \
  --steps 300 \
  --style-weight 1000000 \
  --content-weight 1
```

**Short form:**

```bash
python main.py -c content.jpg -s style.jpg -o output.jpg
```

**Get help:**

```bash
python main.py --help
```

### CLI Arguments

- `--content`, `-c`: Path to content image (required)
- `--style`, `-s`: Path to style image (required)
- `--output`, `-o`: Path to save output image (default: `output.jpg`)
- `--size`: Output image size (default: `512`)
- `--steps`: Number of optimization steps (default: `300`)
- `--style-weight`: Style loss weight (default: `1000000`)
- `--content-weight`: Content loss weight (default: `1`)

## Using stylize_image function

You can use the `stylize_image` function directly in your code:

```python
from src.stylize import stylize_image
from src.utils import save_image

output = stylize_image(
    content_path="assets/content2.jpg",
    style_path="assets/style4.webp",
    output_size=512,
    num_steps=300,
    style_weight=1_000_000,
    content_weight=1
)

save_image(output, "assets/output.jpg")
```

## Parameters

- `content_path`: Path to content image
- `style_path`: Path to style image
- `output_size`: Output image size (default: 512)
- `num_steps`: Number of optimization steps (default: 300)
- `style_weight`: Style loss weight (default: 1_000_000)
- `content_weight`: Content loss weight (default: 1)

## Device Support

The project automatically detects and uses:

- CUDA (NVIDIA GPU) - if available
- MPS (Apple Silicon) - if available
- CPU - otherwise

The device type is printed to console on startup.
