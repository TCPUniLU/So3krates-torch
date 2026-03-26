import torch


class CosineCutoff(torch.nn.Module):
    """
    Equation 17 from so3krates paper
    https://doi.org/10.1038/s41467-024-50620-6
    """

    r_max: torch.Tensor

    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * (torch.cos(torch.pi * x / self.r_max) + 1.0)
            if isinstance(x, torch.Tensor)
            else 0.5 * (torch.cos(torch.pi * x / self.r_max) + 1.0).item()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"


class PhysNetCutoff(torch.nn.Module):
    """
    Cutoff function used in PhysNet.
    """

    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = r_max

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Distances, shape: (...)

        Returns:
            Cutoff fn output with value=0 for r > r_max, shape (...)
        """
        x_norm = r / self.r_max
        # PhysNet cutoff: 1 - 6*x^5 + 15*x^4 - 10*x^3
        cutoff_values = (
            torch.ones_like(r)
            - 6 * x_norm**5
            + 15 * x_norm**4
            - 10 * x_norm**3
        )
        # Apply mask: 0 for r >= r_max, cutoff_values for r < r_max
        return torch.where(r < self.r_max, cutoff_values, torch.zeros_like(r))

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"


class PolynomialCutoff(torch.nn.Module):
    """
    Polynomial cutoff function.
    """

    def __init__(self, r_max: float, p: int):
        super().__init__()
        self.r_max = r_max
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Distances, shape: (...)

        Returns:
            Cutoff fn output with value=0 for r > r_max, shape (...)
        """
        x_norm = r / self.r_max
        p = self.p

        # Polynomial cutoff formula
        cutoff_values = (
            1
            - (1 / 2) * (p + 1) * (p + 2) * x_norm**p
            + p * (p + 2) * x_norm ** (p + 1)
            - (1 / 2) * p * (p + 1) * x_norm ** (p + 2)
        )
        # Apply mask: 0 for r >= r_max, cutoff_values for r < r_max
        return torch.where(r < self.r_max, cutoff_values, torch.zeros_like(r))

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max}, p={self.p})"


class ExponentialCutoff(torch.nn.Module):
    """
    Exponential cutoff function used in SpookyNet.
    """

    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = r_max

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Distances, shape: (...)

        Returns:
            Cutoff fn output with value=0 for r > r_max, shape (...)
        """
        # Exponential cutoff: exp(-r^2 / ((r_max - r) * (r_max + r)))
        #                   = exp(-r^2 / (r_max^2 - r^2))
        mask = r < self.r_max

        # For numerical stability, only compute where r < r_max
        cutoff_values = torch.zeros_like(r)

        # Compute only where mask is True to avoid division issues
        r_masked = torch.where(mask, r, torch.zeros_like(r))
        denominator = self.r_max**2 - r_masked**2

        # Add small epsilon to avoid division by zero at r = r_max
        denominator = torch.where(
            mask,
            torch.clamp(denominator, min=1e-8),
            torch.ones_like(denominator),
        )

        exp_values = torch.exp(-(r_masked**2) / denominator)
        cutoff_values = torch.where(mask, exp_values, torch.zeros_like(r))

        return cutoff_values

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"


cutoff_fn_dict = {
    "cosine": CosineCutoff,
    "phys": PhysNetCutoff,
    "polynomial": PolynomialCutoff,
    "exponential": ExponentialCutoff,
}
