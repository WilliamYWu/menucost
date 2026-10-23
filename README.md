# Monetary menu-cost KS model
Tested on CPU with Python 3.11.5. Open the notebook from this directory so `menucost` is importable.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name menucost-ks --display-name "Menu-cost KS"
```

Select the Menu-cost KS kernel in your notebook editor and run `menucost_walkthrough.ipynb` from top to bottom. Restart an existing kernel before running so the CPU setting takes effect.

The notebook uses small grids for a CPU demonstration. Menu costs use `mp.f` directly. VFI prints every 100 iterations and at its final partial interval; KS prints each pass and its final convergence result.
