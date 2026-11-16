# Reference implementation

`gaussian-splatting` is a Git submodule pinned to the revision in
`graphdeco-commit.txt` from https://github.com/graphdeco-inria/gaussian-splatting.
Studio uses its reference arguments, SH utilities, SSIM and LPIPS-VGG evaluator.

Run `./start.sh` to initialize this dependency automatically. Studio does not
require the upstream SIBR viewer or recursive submodule compilation.

The GraphDeco source retains its original copyright and LICENSE.md, including
its non-commercial research and evaluation terms. Publishing this Studio does
not change the licenses of GraphDeco, gsplat, PyTorch or pretrained weights.
