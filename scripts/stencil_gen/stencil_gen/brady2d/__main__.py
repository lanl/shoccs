"""Allow ``python -m stencil_gen.brady2d`` to invoke the Brady-Livescu 2D CLI."""

import sys

from stencil_gen.brady2d_cli import main

sys.exit(main())
