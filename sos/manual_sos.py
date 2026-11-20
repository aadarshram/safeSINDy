"""
Hand-rolled Putinar-style SOS certificate via cvxpy, bypassing the
SumOfSquares package. Certifies: poly(x) >= 0 for all x in the set
{g_i(x) >= 0 for all i}, by finding SOS multipliers sigma_i such that

    poly(x) - sum_i sigma_i(x) * g_i(x)   is itself SOS.

Each sigma_i (and the leftover SOS term, sigma_0) is parametrized as
z(x)^T Q z(x) for an unknown PSD matrix Q and a fixed monomial basis z(x).
Coefficients are matched exactly via sympy.Poly, giving linear equality
constraints on Q's entries -- no symbolic-numeric bridge beyond that.
"""
import itertools

import cvxpy as cp
import numpy as np
import sympy as sp


def monomial_basis(variables, deg):
    """All monomials in VARIABLES with total degree <= deg."""
    basis = []
    for total in range(deg + 1):
        for combo in itertools.combinations_with_replacement(range(len(variables)), total):
            mono = sp.Integer(1)
            for idx in combo:
                mono *= variables[idx]
            if mono not in basis:
                basis.append(mono)
    return basis


def poly_coeffs(expr, variables):
    """Dict: monomial (as sympy expr in sorted vars) -> numeric coefficient."""
    poly = sp.Poly(sp.expand(expr), *variables)
    coeffs = {}
    for monom, coeff in poly.terms():
        mono_expr = sp.Integer(1)
        for var, power in zip(variables, monom):
            mono_expr *= var**power
        coeffs[mono_expr] = coeffs.get(mono_expr, 0.0) + float(coeff)
    return coeffs


def certify(variables, target_poly, ineqs, deg, verbose=False, ineq_prods=False, solver=cp.CLARABEL, eqs=None):
    """
    Try to certify target_poly >= 0 on {g >= 0 for g in ineqs, h == 0 for h in eqs}, via
        target_poly == sigma_0 + sum_i sigma_i * g_i + sum_j lambda_j * h_j,
        sigma_* SOS, lambda_j a free (not sign-constrained) polynomial.

    Arguments:
        variables: list of sympy symbols.
        target_poly: sympy expression, the polynomial to certify >= 0.
        ineqs: list of sympy expressions g_i, each required >= 0 on the set.
        deg: half the degree bound for each sigma_i's monomial basis
             (so sigma_i has degree <= 2*deg).
        verbose: print solver diagnostics.
        eqs: list of sympy expressions h_j, each required == 0 on the set
             (e.g. the unit-circle constraint cos^2+sin^2-1). Unlike ineqs,
             these get a free-sign polynomial multiplier, not an SOS one --
             an equality can be used in either direction.

    Returns:
        (status, problem). status is 'optimal' (certified) or something
        else (not certified at this degree -- NOT proof of infeasibility).
    """
    eqs = list(eqs) if eqs else []
    basis = monomial_basis(variables, deg)
    n = len(basis)

    multipliers = [sp.Integer(1)] + list(ineqs)
    if ineq_prods:
        for g_i, g_j in itertools.combinations(ineqs, 2):
            multipliers.append(sp.expand(g_i * g_j))

    Qs = []
    constraints = []
    for _ in multipliers:
        Q = cp.Variable((n, n), symmetric=True)
        constraints.append(Q >> 0)
        Qs.append(Q)

    # Free (non-SOS) multiplier for each equality constraint: lambda_j(x) =
    # sum_i c_i * basis[i], c_i a free scalar cvxpy variable.
    lambdas = [cp.Variable(n) for _ in eqs]

    # Build: for each monomial, the linear expression (in Q entries) that
    # must equal the target polynomial's coefficient for that monomial.
    target_coeffs = poly_coeffs(target_poly, variables)
    all_monomials = set(target_coeffs.keys())

    # For each (Q_k, multiplier_k), and each pair (i,j) in the basis,
    # basis[i]*basis[j]*multiplier_k expands to a polynomial; accumulate
    # Q_k[i,j]'s contribution to each resulting monomial's coefficient.
    accum = {}  # monomial -> list of ("Q", k, i, j, numeric_coeff) | ("lam", j, i, numeric_coeff)
    for k, mult in enumerate(multipliers):
        for i in range(n):
            for j in range(n):
                prod = sp.expand(basis[i] * basis[j] * mult)
                prod_coeffs = poly_coeffs(prod, variables)
                for mono, c in prod_coeffs.items():
                    all_monomials.add(mono)
                    accum.setdefault(mono, []).append(("Q", k, i, j, c))

    for eq_idx, h_expr in enumerate(eqs):
        for i in range(n):
            prod = sp.expand(basis[i] * h_expr)
            prod_coeffs = poly_coeffs(prod, variables)
            for mono, c in prod_coeffs.items():
                all_monomials.add(mono)
                accum.setdefault(mono, []).append(("lam", eq_idx, i, c))

    for mono in all_monomials:
        lhs_terms = accum.get(mono, [])
        expr = 0
        for term in lhs_terms:
            if term[0] == "Q":
                _, k, i, j, c = term
                expr = expr + c * Qs[k][i, j]
            else:
                _, eq_idx, i, c = term
                expr = expr + c * lambdas[eq_idx][i]
        rhs = target_coeffs.get(mono, 0.0)
        constraints.append(expr == rhs)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)
        status = prob.status
    except cp.error.SolverError as e:
        status = f"solver_error: {e}"
    return status, prob
