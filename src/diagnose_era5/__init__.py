"""ERA5 preprocessing helpers."""

__all__ = ["create_era5_average"]


def __getattr__(name):
    if name == "create_era5_average":
        from .average import create_era5_average

        return create_era5_average
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
