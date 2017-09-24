# cuda-tiled-matrix-multiplication
**Overview**

Optimized Parallel Tiled Approach to perform Matrix Multiplication by taking advantage of the lower latency, higher bandwidth shared memory within GPU thread blocks.

**Execution**

* Run "make" to build the executable of this file.
* For debugging, run "make dbg=1" to build a debuggable version of the executable binary.
* Run the binary using "./~name-of-the-artifact~"

There are several modes of operation for the application -

* *No arguments*: The application will create two randomly sized and ini-
tialized matrices such that the matrix operation M * N is valid, and P
is properly sized to hold the result. After the device multiplication is in-
voked, it will compute the correct solution matrix using the CPU, and
compare that solution with the device-computed solution. If it matches
(within a certain tolerance), it will print out “Test PASSED” to the screen
before exiting.

* *One argument*: The application will use the random initialization to cre-
ate the input matrices, and write the device-computed output to the file
specified by the argument.

* *Three arguments*: The application will read the input matrices from pro-
vided files. The first argument should be a file containing three integers.
The first, second, and third integers will be used as M.height, M.width
and N.width respectively . The second and third arguments will be ex-
pected to be files which have exactly enough entries to fill matrices M and
N respectively. No output is written to file.

* *Four arguments*: The application will read its inputs from the files pro-
vided by the first three arguments, as described above, and write its output
to the file provided in the fourth.

**Input File Format**

The (optional) input files should have a single line containing whitespace-
separated floating point numbers representing the matrix data. There should
be m · n numbers on this line for a m × n matrix, where the first n numbers are
the first row, the second n numbers are the second row, etc. For example, to
represent the following matrix:
[1 2 3]
[4 5 6]
[7 8 9]
the corresonding input file should contain the following line (without quotes):
“1 2 3 4 5 6 7 8 9”

If you wish to use the output file from one run of the application
as an input in a later run, you must delete the first line in the output file, which
displays the accuracy of the values within the file.