// sbeg/eigen.hpp -- the ONE way the course code includes Eigen.
//
// Why: by default a size mismatch in Eigen (e.g. adding a 3x1 to a 6x1 vector, or reading A(5,0) of a 3x3)
// calls abort() and kills the whole test program, so one bug hides every other test result.
// Here such errors throw sbeg::EigenError instead, which doctest reports as a failed test and then
// continues with the next test. This is active in every build type (Debug and Release).
//
// Always include <sbeg/eigen.hpp> (or a header that includes it: sbeg/labio.hpp, sbeg/mrcpp.hpp)
// BEFORE any <Eigen/...> header.
#ifndef SBEG_EIGEN_HPP
#define SBEG_EIGEN_HPP

#if defined(EIGEN_CORE_H) && !defined(SBEG_EIGEN_ASSERT_THROWS)
#error "Include <sbeg/eigen.hpp> (or sbeg/mrcpp.hpp / sbeg/labio.hpp) before any <Eigen/...> header."
#endif

#include <stdexcept>
#include <string>

namespace sbeg {
/// Thrown instead of abort() when Eigen detects a wrong size or an index out of range.
struct EigenError : std::runtime_error {
    explicit EigenError(const std::string& condition)
        : std::runtime_error("Eigen: wrong matrix size or index out of range (failed check: " +
                             (condition.size() > 100 ? condition.substr(0, 100) + "..." : condition) + ")") {}
};
}  // namespace sbeg

#define SBEG_EIGEN_ASSERT_THROWS 1
#ifndef eigen_assert
#define eigen_assert(x)                                                                                 \
    ((x) ? static_cast<void>(0)                                                                         \
         : throw ::sbeg::EigenError(#x))
#endif

#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif

#include <Eigen/Dense>

namespace sbeg {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}  // namespace sbeg

#endif  // SBEG_EIGEN_HPP
