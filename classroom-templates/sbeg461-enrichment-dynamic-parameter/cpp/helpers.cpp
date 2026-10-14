// SBEG 461 enrichment lab "parameter identification" -- NumPy / SciPy equivalents (C++ track).
// GIVEN code: do not edit.  These replace the library calls of the Python lab, they are not lab content:
//   np.linalg.lstsq -> lstsq, np.linalg.cond -> cond, np.sign -> sign, np.gradient -> gradient,
//   scipy.signal.butter / filtfilt -> butter / filtfilt (same coefficients and edge handling as SciPy),
//   scipy.optimize.least_squares (with jac= and bounds=) -> least_squares (bounded Levenberg-Marquardt).
#include "lab.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lab {
namespace {

using cplx = std::complex<double>;

/// Coefficients of prod_k (x - r_k), highest power first (numpy.poly).
std::vector<cplx> poly(const std::vector<cplx>& roots) {
    std::vector<cplx> c{cplx(1.0, 0.0)};
    for (const cplx& r : roots) {
        std::vector<cplx> d(c.size() + 1, cplx(0.0, 0.0));
        for (std::size_t i = 0; i < c.size(); ++i) {
            d[i] += c[i];
            d[i + 1] -= r * c[i];
        }
        c = d;
    }
    return c;
}

/// scipy.signal.lfilter(b, a, x, zi=zi) (direct form II transposed, a[0] == 1); zi is updated to the final state.
Eigen::VectorXd lfilter(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::VectorXd& x,
                        Eigen::VectorXd& z) {
    const Eigen::Index n = b.size();
    Eigen::VectorXd y(x.size());
    for (Eigen::Index k = 0; k < x.size(); ++k) {
        const double xk = x(k);
        const double yk = b(0) * xk + z(0);
        for (Eigen::Index i = 0; i + 2 < n; ++i) z(i) = b(i + 1) * xk + z(i + 1) - a(i + 1) * yk;
        z(n - 2) = b(n - 1) * xk - a(n - 1) * yk;
        y(k) = yk;
    }
    return y;
}

/// scipy.signal.lfilter_zi(b, a): steady-state initial condition of lfilter for a unit step.
Eigen::VectorXd lfilter_zi(const Eigen::VectorXd& b, const Eigen::VectorXd& a) {
    const Eigen::Index m = b.size() - 1;
    Eigen::MatrixXd IminusA = Eigen::MatrixXd::Identity(m, m);
    // companion(a).T: first COLUMN = -a[1:], super-diagonal ones
    for (Eigen::Index i = 0; i < m; ++i) {
        IminusA(i, 0) += a(i + 1);
        if (i + 1 < m) IminusA(i, i + 1) -= 1.0;
    }
    const Eigen::VectorXd B = b.tail(m) - a.tail(m) * b(0);
    return IminusA.partialPivLu().solve(B);
}

}  // namespace

Eigen::VectorXd lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.size()) throw std::invalid_argument("lstsq: A and b have different numbers of rows");
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(A.rows(), A.cols())));
    return svd.solve(b);
}

double cond(const Eigen::MatrixXd& A) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A);
    const Eigen::VectorXd& s = svd.singularValues();
    if (s.size() == 0) throw std::invalid_argument("cond: empty matrix");
    const double smin = s.minCoeff();
    return smin > 0.0 ? s.maxCoeff() / smin : std::numeric_limits<double>::infinity();
}

Eigen::VectorXd sign(const Eigen::VectorXd& x) {
    Eigen::VectorXd s(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i)
        s(i) = x(i) > 0.0 ? 1.0 : (x(i) < 0.0 ? -1.0 : x(i));  // 0 -> 0, NaN -> NaN (np.sign)
    return s;
}

Eigen::MatrixXd gradient(const Eigen::MatrixXd& X, double h) {
    const Eigen::Index N = X.rows();
    if (N < 2) throw std::invalid_argument("gradient: need at least 2 rows");
    Eigen::MatrixXd D(N, X.cols());
    D.row(0) = (X.row(1) - X.row(0)) / h;
    D.row(N - 1) = (X.row(N - 1) - X.row(N - 2)) / h;
    for (Eigen::Index i = 1; i + 1 < N; ++i) D.row(i) = (X.row(i + 1) - X.row(i - 1)) / (2.0 * h);
    return D;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> butter(int order, double wn) {
    if (order < 1 || !(wn > 0.0 && wn < 1.0)) throw std::invalid_argument("butter: need order >= 1 and 0 < wn < 1");
    // analog prototype (scipy buttap), pre-warp (fs = 2), lp2lp, bilinear transform, zpk -> tf
    const double fs = 2.0, fs2 = 2.0 * fs;
    const double warped = 2.0 * fs * std::tan(PI * wn / fs);
    std::vector<cplx> p;
    for (int m = -order + 1; m < order; m += 2)
        p.push_back(-std::exp(cplx(0.0, PI * static_cast<double>(m) / (2.0 * order))) * warped);
    double k = std::pow(warped, order);
    cplx den(1.0, 0.0);
    std::vector<cplx> pz, zz(static_cast<std::size_t>(order), cplx(-1.0, 0.0));
    for (const cplx& pk : p) {
        pz.push_back((fs2 + pk) / (fs2 - pk));
        den *= (fs2 - pk);
    }
    k *= (cplx(1.0, 0.0) / den).real();
    const std::vector<cplx> bc = poly(zz), ac = poly(pz);
    Eigen::VectorXd b(order + 1), a(order + 1);
    for (int i = 0; i <= order; ++i) {
        b(i) = k * bc[static_cast<std::size_t>(i)].real();
        a(i) = ac[static_cast<std::size_t>(i)].real();
    }
    return {b, a};
}

Eigen::MatrixXd filtfilt(const Eigen::VectorXd& b_in, const Eigen::VectorXd& a_in, const Eigen::MatrixXd& X) {
    const Eigen::Index nb = std::max(b_in.size(), a_in.size());
    if (nb < 2 || a_in(0) == 0.0) throw std::invalid_argument("filtfilt: invalid filter coefficients");
    Eigen::VectorXd b = Eigen::VectorXd::Zero(nb), a = Eigen::VectorXd::Zero(nb);
    b.head(b_in.size()) = b_in / a_in(0);
    a.head(a_in.size()) = a_in / a_in(0);
    const Eigen::Index padlen = 3 * nb, N = X.rows();
    if (N <= padlen) throw std::invalid_argument("filtfilt: the signal must be longer than 3 max(len(a), len(b))");
    const Eigen::VectorXd zi = lfilter_zi(b, a);
    Eigen::MatrixXd Y(N, X.cols());
    for (Eigen::Index c = 0; c < X.cols(); ++c) {
        const Eigen::VectorXd x = X.col(c);
        Eigen::VectorXd ext(N + 2 * padlen);  // odd extension at both ends
        for (Eigen::Index i = 0; i < padlen; ++i) {
            ext(i) = 2.0 * x(0) - x(padlen - i);
            ext(N + padlen + i) = 2.0 * x(N - 1) - x(N - 2 - i);
        }
        ext.segment(padlen, N) = x;
        Eigen::VectorXd z = zi * ext(0);
        Eigen::VectorXd y = lfilter(b, a, ext, z);
        Eigen::VectorXd yr = y.reverse();
        z = zi * yr(0);
        y = lfilter(b, a, yr, z).reverse();
        Y.col(c) = y.segment(padlen, N);
    }
    return Y;
}

LeastSquaresResult least_squares(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& fun,
                                 const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& jac,
                                 const Eigen::VectorXd& x0, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                                 int max_nfev) {
    const Eigen::Index nx = x0.size();
    if (lb.size() != nx || ub.size() != nx) throw std::invalid_argument("least_squares: bounds must have the size of x0");
    if (((x0 - lb).array() <= 0.0).any() || ((ub - x0).array() <= 0.0).any())
        throw std::invalid_argument("least_squares: x0 must be strictly inside the bounds");
    // Like SciPy's 'trf': Coleman-Li scaling d_i = sqrt(distance to the bound the gradient points to), damped
    // Gauss-Newton (Levenberg) steps in the scaled variables, steps truncated to stay strictly inside the bounds,
    // SciPy's default stopping tolerances ftol = xtol = gtol = 1e-8.
    const double ftol = 1e-8, xtol = 1e-8, gtol = 1e-8;
    LeastSquaresResult out;
    Eigen::VectorXd x = x0;
    Eigen::VectorXd r = fun(x);
    out.nfev = 1;
    double cost = 0.5 * r.squaredNorm();
    Eigen::MatrixXd J = jac(x);
    if (J.rows() != r.size() || J.cols() != nx) throw std::invalid_argument("least_squares: jac(x) must be len(r) x len(x)");
    double lam = -1.0;
    while (out.nfev < max_nfev) {
        const Eigen::VectorXd g = J.transpose() * r;
        Eigen::VectorXd d(nx);
        for (Eigen::Index i = 0; i < nx; ++i) {
            const double v = g(i) < 0.0 ? ub(i) - x(i) : (g(i) > 0.0 ? x(i) - lb(i) : 1.0);
            d(i) = std::sqrt(std::isfinite(v) ? v : 1.0);
        }
        const Eigen::VectorXd gs = d.cwiseProduct(g);
        if (gs.cwiseAbs().maxCoeff() < gtol) break;
        const Eigen::MatrixXd Js = J * d.asDiagonal();
        const Eigen::MatrixXd A = Js.transpose() * Js;
        if (lam < 0.0) lam = 1e-3 * std::max(A.diagonal().maxCoeff(), 1e-300);
        bool accepted = false, stop = false;
        while (!accepted && out.nfev < max_nfev) {
            Eigen::MatrixXd H = A;
            H.diagonal().array() += lam;
            Eigen::VectorXd dx = d.cwiseProduct(H.ldlt().solve(-gs));
            double alpha = 1.0;  // largest fraction of the step that stays inside the bounds
            for (Eigen::Index i = 0; i < nx; ++i) {
                if (dx(i) > 0.0) alpha = std::min(alpha, (ub(i) - x(i)) / dx(i));
                if (dx(i) < 0.0) alpha = std::min(alpha, (lb(i) - x(i)) / dx(i));
            }
            if (alpha < 1.0) dx *= 0.995 * alpha;
            const Eigen::VectorXd xn = x + dx;
            const Eigen::VectorXd rn = fun(xn);
            ++out.nfev;
            const double cost_n = 0.5 * rn.squaredNorm();
            if (std::isfinite(cost_n) && cost_n < cost) {
                stop = (cost - cost_n) < ftol * cost || dx.norm() < xtol * (xtol + x.norm());
                x = xn;
                r = rn;
                cost = cost_n;
                lam = std::max(lam / 3.0, 1e-300);
                accepted = true;
            } else {
                lam *= 4.0;
                if (dx.norm() < xtol * (xtol + x.norm()) || !std::isfinite(lam)) {
                    stop = true;
                    break;
                }
            }
        }
        if (stop || !accepted) break;
        J = jac(x);
    }
    out.x = x;
    out.cost = cost;
    return out;
}

}  // namespace lab
