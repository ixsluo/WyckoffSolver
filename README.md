# Wyckoff Combinations

Solve the wyckoff combinations for a give number of elements.

## Installation

```bash
git clone https://github.com/ixsluo/WyckoffSolver.git
cd WyckoffSolver
pip install .
# uninstal
pip uninstall wyckoff_solver
```

## Usage

- From command line

    Get the number of total solutions for number of atoms [8,4,16] in space group 23:

    ```bash
    wyckoffsolver solve 8 4 16 -g 23
    # There are 150840 different solutions
    ```

    Get one solution:

    ```bash
    # get a random one
    wyckoffsolver getone 8 4 16 -g 23
    # or get a specific one ([-i 0] for the first one)
    wyckoffsolver getone 8 4 16 -g 23 -i 0
    ```

- As python lib

    ```python
    from wyckoff_solver import WyckCombSolver

    solver = WyckCombSolver(group=23, num_atoms=[8, 4, 16])
    print(solver.num_solutions)
    # 150840
    print(solver.get_solution(0))
    # array([[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0]], dtype=uint16)
    ```

## Implementation

Let $`\boldsymbol{p}=\{\boldsymbol{p}_i\}`$ be multiplicity of sites which is not variable, i.e. cannot be occupied multiple times.
And $`\boldsymbol{q}=\{\boldsymbol{q}_j\}`$ be multiplicity of sites which is variable, i.e. can be occupied multiple times.
And $`\boldsymbol{u}=\{\boldsymbol{u}_k\}`$ be number of each element.

Let one solutions is formed by $`\hat{\boldsymbol{p}}`$ and $`\hat{\boldsymbol{q}}`$, which are occupied times on each site categoried by variability, respectively.

To reduce number of variable, sites with same multiplicity in $`\boldsymbol{b}`$ can be merged together.

The solutions should satisfies:

```math
\begin{cases}
\boldsymbol{p}_i \hat{\boldsymbol{p}}_{ki} + \boldsymbol{q}_j \hat{\boldsymbol{q}}_{kj} = \boldsymbol{u}_k, \quad k=0,1,\cdots \\
0 \le \hat{\boldsymbol{p}}_{ki} \le 1, \quad k,i=0,1,\cdots \\
\sum_{k}\hat{\boldsymbol{p}}_{ki} \le 1, \quad k=0,1,\cdots \\
\hat{\boldsymbol{q}}_{kj} \ge 0 \quad k,j=0,1,\cdots\ .
\end{cases}
```

-----------

Our discussion begins from $`\boldsymbol{p}`$. Let $`\beta=\gcd(\boldsymbol{q})`$, immediately we have

```math
\beta | \boldsymbol{u}_k - \boldsymbol{p}_i\hat{\boldsymbol{p}}_{ki}, \quad k=0,1,\cdots \ .
```

This is called an **Exclusive Binary Divisible Linear Programming problem**. All $`\hat{\boldsymbol{p}}`$ can be easily solved.

-----------

Then, we solve the corresponding solutions $`\hat{\boldsymbol{q}}`$ for each $`\hat{\boldsymbol{p}}`$.

Denote $`\boldsymbol{u}^p:=\boldsymbol{p}_i\hat{\boldsymbol{p}}_{ki}`$ and $`\boldsymbol{u}^q:=\boldsymbol{u} - \boldsymbol{u}^p`$. Take one solution of $`\hat{\boldsymbol{p}}`$, then $`\boldsymbol{u}^q`$ is known. The constrains become

```math
\begin{cases}
\boldsymbol{q}_j\hat{\boldsymbol{q}}_{kj} = \boldsymbol{u}^q_k, \quad k=0,1,\cdots \\
\hat{\boldsymbol{q}}_{kj} \ge 0 \quad k,j=0,1,\cdots\ .
\end{cases}
```

We can reduce $`\boldsymbol{q}_j`$ and $`\boldsymbol{u}^q_k`$ by $`\beta`$. These will not change the solutions and the form of the equations. For simplarity, we collect all $`\boldsymbol{q}_j`$ which have the same value to get the solution of total values $`\tilde{\boldsymbol{q}}_{kj}`$ of them. Use combination to unpack them latter.

Denote the reduced and collected coefficients are $`\boldsymbol{q}^{\prime}_j`$ and $`\boldsymbol{u}^{\prime q}_k`$. The constrains become

```math
\begin{cases}
\boldsymbol{q}^{\prime}_j\tilde{\boldsymbol{q}}_{kj} = \boldsymbol{u}^{\prime q}_k, \quad k=0,1,... \\
\tilde{\boldsymbol{q}}_{kj} \ge 0 \quad k,j=0,1,\cdots \ .
\end{cases}
```

These are $`k`$ independent **Non-Negative Linear Diophantine Problems**.

The generation function is

```math
G(x) = \prod_{j}(1 + x^{\boldsymbol{q}^{\prime}_j} + x^{2\boldsymbol{q}^{\prime}_j} + \cdots)
```

the number of sulutions for $`\boldsymbol{u}^{\prime q}_{k}`$ equals to the coefficients of term $`x^{\boldsymbol{u}^{\prime q}_{k}}`$ in $`G(x)`$.

Using dynamic programming method to count the number of solutions, and backtracing method to find all solutions.

Note: use reversed-sorted $`\boldsymbol{q}^{\prime}`$ can accelerate computation.

-----------
Finally, for each collected solution $`\tilde{\boldsymbol{q}}_{kj}`$, unpacking them falls back to a series of **Integer Partition Problems**.

Assume the original $`\boldsymbol{q}_j`$ has $`v`$ repeated times. The solution $`\hat{\boldsymbol{q}}_{kj}`$ divides into $`\{\hat{\boldsymbol{q}}_{kjl}|l=1,\cdots,v\}`$. It satisfies

```math
\begin{cases}
\sum_{l=1}^v{\hat{\boldsymbol{q}}_{kjl}} = \tilde{\boldsymbol{q}}_{kj} \\
\hat{\boldsymbol{q}}_{kjl} \ge 0 \quad k,j=0,1,\cdots \ .
\end{cases}
```

It also is a **Non-Negative Linear Diophantine Problem** with all coeficients are 1. The generation function

```math
G(x) = (1 + x + x^2 + ... + x^{\tilde{\boldsymbol{q}}_{kj}})^v
```

the number of solutions is $`\text{C}_{v - 1 + \tilde{\boldsymbol{q}}_{kj}}^{v-1}`$.

Also use backtracking method to get all solutions.
