# CursorMix checks

Install the assistant dependencies from the [user guide](../../docs/USER_GUIDE.md#cursormix-for-macos). On a logged-in Mac, quit any existing copy before running a check that claims the global shortcut.

```sh
.venv-clicky/bin/python -m clicky --check-startup
.venv-clicky/bin/python -m eval.clicky.corpus --install router agent
```

To install the public screen fixtures, run the following from the project root. The installers check recorded sources and place downloaded fixtures under ignored `data/` paths.

```sh
.venv-clicky/bin/python -m eval.clicky.corpus --install latency qa
uv run --no-project --python 3.12 --with pyarrow --with pillow eval/clicky/screen2ax.py
.venv-clicky/bin/python -m eval.clicky.corpus --install point
.venv-clicky/bin/python eval/clicky/run_all.py
```

The public router set covers negative cases. The full gate also needs positive desktop requests. The public pointing set covers Retina fixtures. Native 1× pointing, teaching, and paired desktop action gates need their own recordings. The suite leaves these gates incomplete when the evidence is missing.

For a manual trial, open a disposable Finder window, press **Control–Option–Space**, and ask about a visible control. Check that the response describes the foreground window, the pointer lands on the requested control, and Escape restores focus. Save any further observations locally. See [recorded aggregate results](results.md).
