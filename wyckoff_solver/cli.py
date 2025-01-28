import argparse
import textwrap
import time

from wyckoff_solver.wyckcomb import WyckCombSolver

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Wyckoff Combination Problem

            This tool is designed to solve the wyckoff combination problem.
            The symmetry operations of the space group is used to generate
            all possible combinations for atoms with given target numbers
            The sotutions are cached to avoid recalculating the same combinations.
            The order of the target numbers won't affect the cache.

            Action:
                solve   solve the wyckoff combination problem
                        and return the number of solutions
                getone  get a one solution from the solution set;
                        if [-i] is not specified, a random solution is returned;
                        if no solution is available, exit with returncode 1

            Notes:
                The solution set can be very large and may take a long time to solve
                for large target numbers

        """),
    )
    parser.add_argument('action', choices=["solve", "getone"], help="action to do; see docs above")
    parser.add_argument('targets', nargs='+', type=int, help="number of atoms of each species")
    parser.add_argument('-g', '--group-number', metavar='G', type=int, default=1, help="group number (default: 1)")
    parser.add_argument('-i', '--index', metavar='i', type=int, help="index of which solution to get (default: None)")

    args = parser.parse_args()
    return args


def main():
    parsed_args = parse_args()

    solver = WyckCombSolver(
        group=parsed_args.group_number,
        num_atoms=parsed_args.targets,
    )

    if parsed_args.action == 'solve':
        print(f'Solving {parsed_args.targets} in SpaceGroup {solver.group.number} ...')
        t_start = time.time()
        cachefile = solver.descending_solver.cachefile
        if cachefile is not None and cachefile.exists():
            print("Cache loaded")
        num_sol = solver.num_solutions
        t_stop = time.time()
        print(f"Time used: {t_stop - t_start:.6f} seconds")
        print(f"Number of total solutions: {num_sol}")
    elif parsed_args.action == 'getone':
        if solver.has_solutions:
            import numpy as np

            if parsed_args.index is None:
                import random

                idx = random.randint(0, solver.num_solutions)
            else:
                idx = parsed_args.index
            sol = solver.get_solution(idx)

            count_decimal_digits = np.vectorize(lambda num: len(str(num)))
            target_width = max(np.max(count_decimal_digits(parsed_args.targets)), 6)
            header_widths = count_decimal_digits(solver.group.multiplicity) + 1
            body_widths = np.max(count_decimal_digits(sol), axis=0)
            widths = np.maximum(header_widths, body_widths)
            header_fmt = ' '.join(f"{{:{w}s}}" for w in widths)
            body_fmt = ' '.join(f"{{:{w}d}}" for w in widths)

            header = [
                f'{mul}{letter}'
                for mul, letter in zip(solver.group.multiplicity, solver.group.letters)
            ]
            print(f'{{:{target_width}}}'.format('target'), header_fmt.format(*header))
            for target, row in zip(parsed_args.targets, sol):
                print(f'{{:{target_width}}}'.format(target), body_fmt.format(*row))

        else:
            import sys

            print(
                f'No solutions for {parsed_args.targets} SpaceGroup {solver.group.number} ...',
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        raise ValueError(f"Invalid action: {parsed_args.action}")


if __name__ == "__main__":
    main()
