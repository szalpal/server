#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import glob
import itertools
import re
import argparse
import sys
import subprocess

# Linter script, that calls cpplint.py, specifically for DALI repo.
# This will be called in `make lint` cmake target

# Q: How to configure git hook for pre-push linter check?
# A: Create a file `.git/hooks/pre-push`:
#
# #!/bin/sh
# DALI_ROOT_DIR=$(git rev-parse --show-toplevel)
# python $DALI_ROOT_DIR/tools/lint.py $DALI_ROOT_DIR --nproc=10
# ret=$?
# if [ $ret -ne 0 ]; then
#     exit 1
# fi
# exit 0


# Specifies, which files are to be excluded
# These filters are regexes, not typical unix-like path specification
negative_filters = [
    ".*utils/logging.h",
    ".*utils/logging.cc",
]


def negative_filtering(patterns: list, file_list):
    """
    Patterns shall be a list of regex patterns
    """
    if len(patterns) == 0:
        return file_list
    prog = re.compile(patterns.pop())
    it = (i for i in file_list if not prog.search(i))
    return negative_filtering(patterns, it)


def gather_files(path: str, patterns: list, antipatterns: list):
    """
    Gather files, based on `path`, that match `patterns` unix-like specification
    and do not match `antipatterns` regexes
    """
    curr_path = os.getcwd()
    os.chdir(path)
    positive_iterators = [glob.iglob(os.path.join('**', pattern), recursive=True) for pattern in
                          patterns]
    linted_files = itertools.chain(*positive_iterators)
    linted_files = (os.path.join(path, file) for file in linted_files)
    linted_files = negative_filtering(antipatterns.copy(), linted_files)
    ret = list(linted_files)
    os.chdir(curr_path)
    return ret


def gen_cmd(root_dir, file_list, process_includes=False):
    """
    Command for calling cpplint.py
    """
    cmd = ["python",
           os.path.join(root_dir, "extern", "cpplint.py"),
           "--quiet",
           "--linelength=100",
           "--root=" + os.path.join(root_dir, "include" if process_includes else "")]
    cmd.extend(file_list)
    return cmd


def lint(root_dir, file_list, process_includes, n_subproc):
    """
    n_subprocesses: how many subprocesses to use for linter processing
    Returns: 0 if lint passed, 1 otherwise
    """
    if len(file_list) == 0:
        return 0
    cmds = []
    diff = int(len(file_list) / n_subproc)
    for process_idx in range(n_subproc - 1):
        cmds.append(gen_cmd(root_dir=root_dir,
                            file_list=file_list[process_idx * diff: (process_idx + 1) * diff],
                            process_includes=process_includes))
    cmds.append(gen_cmd(root_dir=root_dir,
                        file_list=file_list[(n_subproc - 1) * diff:],
                        process_includes=process_includes))
    subprocesses = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in
                    cmds]
    success = True
    for subproc in subprocesses:
        stdout, stderr = subproc.communicate()
        success *= not bool(subproc.poll())
        if len(stderr) > 0:
            print(stderr.decode("utf-8"))
    return 0 if success else 1


def main(root_dir, n_subproc=1, file_list=None):
    cc_files = gather_files(
        os.path.join(root_dir, "dali_backend"),
        ["*.cc", "*.h", "*.cu", "*.cuh"] if file_list is None else file_list,
        negative_filters)

    cc_code = lint(root_dir=root_dir, file_list=cc_files, process_includes=False,
                   n_subproc=n_subproc)

    if cc_code != 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run linter check for project files. "
                                                 "Gather all code-files (h, cuh, cc, cu, inc, inl) and perform linter check on them.")
    parser.add_argument('root_path', type=str,
                        help='Root path of the repository (pointed directory should contain `.git` folder)')
    parser.add_argument('--nproc', type=int, default=1,
                        help='Number of processes to spawn for linter verification')
    parser.add_argument('--file-list', nargs='*',
                        help='List of files. This overrides the default scenario')
    args = parser.parse_args()
    assert args.nproc > 0
    main(str(args.root_path), args.nproc, file_list=args.file_list)
