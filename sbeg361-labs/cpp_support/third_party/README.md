# Third-party libraries (vendored, unmodified)

| Library | Version | Licence | Files |
|---|---|---|---|
| Eigen | 3.4.0 | MPL-2.0 (`eigen/COPYING.MPL2`) | `eigen/Eigen/` (headers only) |
| doctest | 2.4.11 | MIT (`doctest/LICENSE.txt`) | `doctest/doctest.h` |
| JSON for Modern C++ (nlohmann/json) | 3.11.3 | MIT (`nlohmann/LICENSE.MIT`) | `nlohmann/json.hpp` |

The sources are unmodified copies of the upstream releases. The course build defines `EIGEN_MPL2_ONLY`, so only
MPL-2.0-licensed parts of Eigen can be compiled. The MPL-2.0 permits redistribution of these unmodified files;
their source is available at https://eigen.tuxfamily.org, https://github.com/doctest/doctest and
https://github.com/nlohmann/json.
