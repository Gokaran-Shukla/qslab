"""Convert tutorials/src/*.py (jupytext percent format) into executed notebooks."""
import sys, os, glob, time
import jupytext, nbformat
from nbclient import NotebookClient

root = os.path.join(os.path.dirname(__file__), "..", "tutorials")
srcs = sorted(glob.glob(os.path.join(root, "src", "*.py")))
if len(sys.argv) > 1:
    srcs = [s for s in srcs if any(k in s for k in sys.argv[1:])]
for src in srcs:
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = os.path.join(root, os.path.basename(src).replace(".py", ".ipynb"))
    t = time.time()
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": root}}).execute()
        status = "ok"
    except Exception as e:
        status = f"FAILED: {type(e).__name__}: {str(e)[-1500:]}"
    nbformat.write(nb, out)
    print(f"{os.path.basename(out)}: {status} ({time.time()-t:.0f}s)", flush=True)
