# Hello

**This is an example tool to be removed in the future.**

Execute `frame hello -h` for help about this tool.

To remove this tool, delete `tools/hello` and `tests/hello`, and supress `'tools.hello'` from the
`PACKAGES` list in `setup.py`, and `"hello": tools.hello.hello.main` from the `TOOLS` dictionary
in `tools/frame.py`. Finally, re-execute `pip install -e .` from the top-level project folder.