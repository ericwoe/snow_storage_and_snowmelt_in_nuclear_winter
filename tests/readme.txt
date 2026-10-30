Unit tests for the snow model (src/processing/snow_model.py), run with pytest
from the repository root:

    pytest

They also run automatically on every push via .github/workflows/testing.yml.

test_snow_model.py
    test_output_shape                Output arrays keep the (time, lat, lon) shape.
    test_melt_never_exceeds_storage  Melt is limited by the available snow storage,
                                     and storage never becomes negative.
    test_degree_day_formula          Melt follows the degree-day formula
                                     (days x degree-day factor x (T - T_melt)),
                                     capped at the available snow.
    test_spatial_independence        Grid cells are computed independently
                                     (cold cell accumulates snow, warm cell not).
    test_storage_mass_balance        Storage(t) = storage(t-1) + new snow - melt.
