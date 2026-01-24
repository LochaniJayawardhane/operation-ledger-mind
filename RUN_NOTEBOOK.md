# How to Run the Data Factory Notebook

## Prerequisites

✅ Dependencies are already installed via `uv sync`

## Step 1: Set up Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Step 2: Run the Notebook

You have several options:

### Option A: Using Jupyter (Recommended)

1. **Activate the virtual environment** (if not already active):
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Start Jupyter**:
   ```powershell
   uv run jupyter notebook
   ```
   Or use JupyterLab:
   ```powershell
   uv run jupyter lab
   ```

3. Navigate to `notebooks/01_data_factory.ipynb` and open it

4. Run all cells (Cell → Run All) or run cells one by one

### Option B: Using VS Code

1. **Open the project** in VS Code

2. **Select the Python interpreter**:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.venv` (should show `Python 3.x.x ('.venv': venv)`)

3. **Open the notebook**:
   - Navigate to `notebooks/01_data_factory.ipynb`
   - VS Code will automatically detect it as a Jupyter notebook

4. **Run cells**:
   - Click "Run All" at the top, or
   - Use the play button on each cell

### Option C: Using uv run directly

```powershell
uv run jupyter notebook notebooks/01_data_factory.ipynb
```

## Step 3: Verify Setup

Before running the full pipeline, you can verify:

1. **Check API key is loaded**: The first cell should print "✓ OpenAI API key loaded: Yes"
2. **Check PDF path**: The configuration cell should show the correct PDF path
3. **Check paths exist**: All paths should show "exists: True"

## Troubleshooting

### If you get "OPENAI_API_KEY not found":
- Make sure `.env` file exists in the project root
- Make sure it contains: `OPENAI_API_KEY=your_key_here`
- No quotes needed around the key

### If you get import errors:
- Make sure you're using the virtual environment: `.venv\Scripts\Activate.ps1`
- Or use `uv run` prefix for commands

### If PDF not found:
- The notebook will automatically find `2024-Annual-Report.pdf` in `data/raw/`
- If you see a warning, it means it found a different PDF file (which is fine)

## Expected Runtime

- **PDF Loading**: ~10-30 seconds (depending on PDF size)
- **Chunking**: Instant
- **Q/A Generation**: ~2-5 minutes per chunk (depends on API rate limits)
  - For a typical annual report with ~100 chunks: ~3-8 hours total
  - Consider running a subset first to test

## Output Files

After completion, you'll find:
- `data/output/train.jsonl` - Training dataset (80% of Q/A pairs)
- `data/output/golden_test_set.jsonl` - Test dataset (20% of Q/A pairs)
