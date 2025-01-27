# Wyckoff Combinations

Let $\bm{p}=(\bm p_i)$ be repeatabilities of sites not repeatable,
and $\bm{q}=(\bm q_j)$ be repeatabilities of sites repeatable,
and $\bm{u}=(\bm u_k)$ be number of each element.

Let $\hat{\bm{p}}$ and $\hat{\bm{q}}$ are solves.

To reduce number of variable, sites with same multiplicity in $\bm{b}$ can be merged together.

Equations:
$$
\bm{p}_i\hat{\bm{p}}_{ki} + \bm{q}_j\hat{\bm{q}}_{kj} = \bm u_k, \quad k=0,1,... \\
0 \le \hat{\bm{p}}_{ki} \le 1, \quad k,i=0,1,... \\
\sum_{k}\hat{\bm{p}}_{ki} \le 1, \quad j=0,1,... \\
\hat{\bm{q}}_{kj} \ge 0 \quad k,j=0,1,... \\
$$

-----------

Discuss $\bm{p}$ first. Let $\beta=\text{gcd}(\bm{q})$,

$$
\beta | \bm{u}_k - \bm{p}_i\hat{\bm{p}}_{ki}, \quad k=0,1,... \\
0 \le \hat{\bm{p}}_{ki} \le 1, \quad k,i=0,1,... \\
\sum_{k}\hat{\bm{p}}_{ki} \le 1, \quad j=0,1,... \\
$$

calling exclusive binary divisible linear programming problem,

we can easily solve all $\hat{\bm{p}}$. We note $\bm{u}^p=(\bm{p}_i\hat{\bm{p}}_{ki})$, and $\bm{u}^q=\bm{u} - \bm{u}^p$

----------

Then discuss $\bm{q}$, the constrains becomes

$$
\bm{q}_j\hat{\bm{q}}_{kj} = \bm{u}^b_k, \quad k=0,1,...\\
\hat{\bm{q}}_{kj} \ge 0, \quad k,j=0,1,... \\
$$

For simplarity, we can collect all $\bm{q}_j$ which have the same value,
then reduce $\bm{q}_j$ and $\bm{u}^q_k$ by $\beta$, resulting in $\bm{q}^{*}_j$ and $\bm{u}^{*q}_k$.

This operation will not change the solutions and the form of the equations.

The constrains become

$$
\bm{q}^{*}_j\hat{\bm{q}}^{*}_{kj} = \bm{u}^{*q}_k, \quad k=0,1,...\\
\hat{\bm{q}}^{*}_{kj} \ge 0, \quad k,j=0,1,... \\
\bm{q}^{*}_0 > \bm{q}^{*}_1 > ... \\
$$

these are $k$ independent non-negative linear diophantine problems.

The generation function is

$$
G(x) = \prod_{j} (1 + x^{\bm{q}^{*}_j} + x^{2\bm{q}^{*}_j} + ...) \
$$

the number of sulutions for $\bm{u}^{*q}_{k}$ equals to the coefficients of term $x^{\bm{u}^{*q}_{k}}$ in $G(x)$.

Using dynamic programming method to count the number of solutions, and backtracing method to find all solutions.

Note that reversed-sorted $\bm{q}^{*}$ can improve performance.

---------
Last, for each solutions, unpacking the collected same values by each solution term falls back to a series of integer partition problems.

Assume the original $\bm{q}_j$ has $v$ repeated times, the target constant term is $\hat{\bm{q}}^{*}_{kj}$.

The equation is

$$
\sum_{l=1}^v{\hat{\bm{q}}_{kjl}} = \hat{\bm{q}}^{*}_{kj} \\
\hat{\bm{q}}_{kjl} \ge 0 \\
$$

It also is a non-negative linear diophantine equation with all coeficients are 1. The generation function

$$
G(x) = (1 + x + x^2 + ... + x^{\hat{\bm{q}}^{*}_{kj}})^v
$$

the number of solutions is $\text{C}_{v - 1 + \hat{\bm{q}}^{*}_{kj}}^{v-1}$.

Also use backtracking method to get all solutions.