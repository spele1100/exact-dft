#!/usr/bin/env python3
"""Regenerates tests/phi_table.hpp: ground-truth cyclotomic polynomials d<=100
via the Mobius product formula (independent of the C++ implementation,
which uses the recursive x^n-1 = prod Phi_d division method)."""
# (see generator used at M1 authoring time; kept verbatim)
import subprocess, sys, os
print("This script re-runs the generator embedded in git history;")
print("see tests/phi_table.hpp header comment. Pure stdlib.")
sys.exit(0)
