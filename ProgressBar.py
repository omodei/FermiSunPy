#!/usr/bin/env python
# Shebang: allows the script to be run directly from the command line
# using the system's default Python interpreter.

import sys  # Used for command-line arguments and direct stdout writing

class progressbar:
    """
    Simple text-based progress bar that prints '*' characters
    as progress advances.
    """

    def __init__(self, toolbar_width=10):
        """
        Initialize the progress bar.

        Parameters
        ----------
        toolbar_width : int
            Total number of '*' characters that represent 100% progress.
        """
        self.toolbar_width = toolbar_width

        # Print an empty progress bar: [          ]
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()

        # Move the cursor back to just after '['
        # so we can overwrite spaces with '*'
        sys.stdout.write("\b" * (toolbar_width + 1))

    def go(self, i, n):
        """
        Update the progress bar for iteration i out of n total steps.

        Parameters
        ----------
        i : int
            Current iteration index (starting from 0).
        n : int
            Total number of iterations.
        """

        # Number of iterations corresponding to one '*'
        x = int(n / self.toolbar_width)

        # If total iterations are fewer than toolbar width,
        # print a '*' at every step
        if x == 0:
            sys.stdout.write("*")
            sys.stdout.flush()

        # Otherwise, print a '*' every x iterations
        elif (i % x) == 0:
            sys.stdout.write("*")
            sys.stdout.flush()

        # When finished, move to the next line
        if i == (n - 1):
            sys.stdout.write("\n")


if __name__ == '__main__':
    # Create a progress bar with width 10
    pb = progressbar(10)

    # Try to read total number of iterations from command-line argument
    try:
        n = int(sys.argv[1])
    except:
        # Default number of iterations if none provided
        n = 100000

    # Simulate a loop with progress updates
    for i in range(n):
        pb.go(i, n)
    
            
