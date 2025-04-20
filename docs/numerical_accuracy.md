# Numerical accuracy

The process of partial fraction decomposition for a rational function expressed as a ratio of polynomials depends on two numerical algorithms:

* root finding for the denominator polynomial $Q(x)$
* solution of a system of linear equations to determine the coefficients

Both of the algorithms will be subject to some amount of numerical noise due to the inherent limits of floating point arithmetic. This is particularly impactful when dealing with polynomials whose roots cover a large range. There is currently no way around this issue in this library (though in theory they could be ameliorated, at the expense of a significant drop in performance, by using higher precision floating point numbers and the corresponding implementations of the algorithms). However, in order to make things a bit easier and cleaner, this library supports in various points the use of numerical tolerances and checks to try to correct these errors. In particular:

* absolute and relative tolerances `atol` and `rtol` can be used in root-finding to make close roots be the same; this uses NumPy's `isclose` functionality. This is meant to compensate for the errors in root-finding that might accidentally split up a root with higher multiplicity;

* a zeroing tolerance `ztol` can be used to reduce any real or imaginary part that's small enough in absolute value to zero. This is meant to correct for roots and coefficients that are supposed to be purely real or imaginary appearing to be complex.

These tolerances appear explicitly and can be set manually in the `RationalFunction` constructor as well as several other methods. When they are explicit arguments, they can be provided and will be used only in the scope of that specific function. However, this is not always possible; these tolerances will be used implicitly also in the operations of product and division. For this reason, it is possible to also set some global values for the tolerances, that will be used by default whenever the user does not provide any. These global defaults are all set to 0 initially (so that no tolerances are applied and all results are unchanged), but can be set by calling the method [`.set_approximation_options`][rational_functions.ratfunc.RationalFunction.set_approximation_options].
