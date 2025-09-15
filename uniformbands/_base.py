import numpy as np

from ._methods import student, uniform


def get_bands(
    F_lo: np.ndarray,
    F_hi: np.ndarray | None = None,
    alpha: float = 0.05,
    *,
    eps: float = 1e-8,
    method: str = "uniform",
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> dict[str, np.ndarray]:
    """Gets the uniform bands according the specified method.

    Args:
        F_lo (np.ndarray): The lower high probability bounds.
        F_hi (np.ndarray):  The upper high probability bounds. Defaults to None.
        alpha (float, optional): The level of supplementary risk. Defaults to 0.05.
        eps (float, optional): The regularization parameter to ensure a well defined
            division. Defaults to 1e-8.
        method (str, optional): Either "uniform" or "student", the method used to
            compute uniform bands. Defaults to "uniform".
        min_val (float, optional): The minimum accepted values of the functions.
            Defaults to 0.0.
        max_val (float, optional): The maximum accepted values of the functions.
            Defaults to 1.0.

    Raises:
        ValueError: If the shapes of F_lo and F_hi don't match.
        ValueError: If F_lo and F_hi do not have at least 2 dimensions.
        ValueError: If alpha is not in (0, 1).
        ValueError: If eps is not strictly positive for 'student' method.
        ValueError: If the method is not 'uniform' nor 'student'.

    Returns:
        dict[str, np.ndarray]: The computed bands with keys "lower" and "upper".
    """
    F_hi = F_hi if F_hi is not None else F_lo

    if F_lo.shape != F_hi.shape:
        raise ValueError(
            f"F_lo and F_hi shapes must be the same, got {F_lo.shape} != {F_hi.shape}"
        )

    if F_lo.ndim < 2:  # noqa: PLR2004
        raise ValueError(
            f"F_lo and F_hi must have at least two dimensions, got shape {F_lo.shape}"
        )

    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    match method:
        case "uniform":
            return uniform(F_lo, F_hi, alpha)
        case "student":
            if eps <= 0:
                raise ValueError(
                    f"eps must be strictly positive for the student method, got {eps}"
                )
            if min_val >= max_val:
                raise ValueError(
                    f"min_val must be less than max_val, got {min_val} >= {max_val}"
                )
            return student(F_lo, F_hi, alpha, eps, min_val, max_val)
        case _:
            raise ValueError(
                f"Method should be in ('uniform', 'student'), got {method}"
            )
