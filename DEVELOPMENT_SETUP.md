# vAIn P2P AGI Development Setup

## Prerequisites
- Python 3.7 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vAIn_p2p_AGI.git
cd vAIn_p2p_AGI
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For development, install in editable mode:
```bash
pip install -e .
```

## Running the application

To run the application in interactive mode:
```bash
python main.py --interactive
```

## Troubleshooting

If you encounter a "ModuleNotFoundError", make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For specific module errors (like "No module named 'backoff'"), you can install individual packages:
```bash
pip install backoff
```
