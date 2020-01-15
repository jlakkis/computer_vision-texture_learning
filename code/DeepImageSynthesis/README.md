# DeepImageSynthesis

This directory is pulled directly from commit 289dfe6fcfe365d3b0aa4fe4445118d28c8f64ec of https://github.com/leongatys/DeepTextures.
Almost none of the code here was written by/belongs to our project group. 

## Modifications
In a few places we had to modify the source code to make it compatible with python 3.7. These include:

- In line 26 of Misc.py, `net.blobs.keys()` is replaced by `list(net.blobs.keys())` to allow indexing
