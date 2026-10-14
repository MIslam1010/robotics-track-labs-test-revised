// sbeg/labio.hpp -- input/output and checking helpers for the SBEG 361 / 461 / 46X C++ labs (header only).
//
//   sbeg::lab_path("cpp_cases.json")              file next to the lab's CMakeLists.txt (the lab/cpp folder)
//   sbeg::lab_path("../data/tracker.csv")         the lab's data folder
//   sbeg::load_json(path)                          -> nlohmann::json (clear error if missing / invalid)
//   sbeg::as_vec(j), sbeg::as_mat(j), sbeg::as_mats(j)       JSON arrays -> Eigen (dynamic size)
//   sbeg::as<Eigen::Matrix3d>(j), sbeg::as_list<Eigen::Matrix4d>(j)   ... fixed size, size checked
//   sbeg::to_json(eigen_object)                    Eigen -> JSON array (vector: flat list, matrix: list of rows)
//   sbeg::read_csv(path)                           -> {header, data} (numeric CSV, first line = header)
//   sbeg::write_csv(path, header, data)            write results for plotting in Python
//   CHECK_MAT_NEAR(got, expected, tol)             doctest check with a readable message (max error, both values)
//   REQUIRE_MAT_NEAR(got, expected, tol)           same, but stops the test case on failure
//   CHECK_SCALAR_NEAR(got, expected, tol)          the same for two doubles (absolute tolerance)
//   CHECK_APPROX(got, expected)                    like Python `got == pytest.approx(expected)` (rel 1e-6, abs 1e-12)
//
// JSON layout used by every lab (cpp_cases.json and data/*.json):
//   vector  -> flat list            [1.0, 2.0, 3.0]
//   matrix  -> list of rows         [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
//   stack   -> list of matrices     [[[...], ...], [[...], ...]]          (Python array of shape (N, r, c))
//   boolean -> true / false,  integer -> number,  NaN/inf are not allowed (JSON has no NaN).
#ifndef SBEG_LABIO_HPP
#define SBEG_LABIO_HPP

#include <sbeg/eigen.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SBEG_LAB_DIR
#define SBEG_LAB_DIR "."
#endif

namespace sbeg {

using json = nlohmann::json;

// ------------------------------------------------------------------ paths and files
/// Path of a file relative to the lab's C++ folder (where CMakeLists.txt is), independent of the working directory.
inline std::string lab_path(const std::string& relative) {
    return std::string(SBEG_LAB_DIR) + "/" + relative;
}

/// UTF-8 string -> filesystem path (folder names may contain non-ASCII characters on Windows).
inline std::filesystem::path fs_path(const std::string& utf8) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
#else
    return std::filesystem::u8path(utf8);
#endif
}

inline std::ifstream open_input(const std::string& path) {
    std::ifstream in(fs_path(path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    return in;
}

/// Load a JSON file; throws std::runtime_error with the path if it is missing or not valid JSON.
inline json load_json(const std::string& path) {
    std::ifstream in = open_input(path);
    try {
        return json::parse(in);
    } catch (const json::exception& e) {
        throw std::runtime_error("invalid JSON in " + path + ": " + e.what());
    }
}

// ------------------------------------------------------------------ JSON -> Eigen
inline double as_double(const json& j) {
    if (!j.is_number()) throw std::runtime_error("JSON value is not a number: " + j.dump());
    return j.get<double>();
}

/// Flat JSON list -> VectorXd. A list of one-element lists ([[1],[2]]) is also accepted.
inline Eigen::VectorXd as_vec(const json& j) {
    if (!j.is_array()) throw std::runtime_error("expected a JSON list for a vector, got: " + j.dump());
    Eigen::VectorXd v(static_cast<Eigen::Index>(j.size()));
    for (std::size_t i = 0; i < j.size(); ++i) {
        const json& e = (j[i].is_array() && j[i].size() == 1) ? j[i][0] : j[i];
        v(static_cast<Eigen::Index>(i)) = as_double(e);
    }
    return v;
}

/// List of rows -> MatrixXd (all rows must have the same length). An empty list gives a 0x0 matrix.
inline Eigen::MatrixXd as_mat(const json& j) {
    if (!j.is_array()) throw std::runtime_error("expected a JSON list of rows for a matrix, got: " + j.dump());
    if (j.empty()) return Eigen::MatrixXd(0, 0);
    if (!j[0].is_array()) throw std::runtime_error("expected a list of rows (a matrix), got a flat list: " + j.dump());
    const std::size_t r = j.size(), c = j[0].size();
    Eigen::MatrixXd m(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
    for (std::size_t i = 0; i < r; ++i) {
        if (!j[i].is_array() || j[i].size() != c) throw std::runtime_error("ragged matrix in JSON: " + j.dump());
        for (std::size_t k = 0; k < c; ++k)
            m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = as_double(j[i][k]);
    }
    return m;
}

/// List of matrices (Python array of shape (N, r, c)) -> std::vector<MatrixXd>.
inline std::vector<Eigen::MatrixXd> as_mats(const json& j) {
    if (!j.is_array()) throw std::runtime_error("expected a JSON list of matrices, got: " + j.dump());
    std::vector<Eigen::MatrixXd> out;
    out.reserve(j.size());
    for (const auto& e : j) out.push_back(as_mat(e));
    return out;
}

/// JSON -> a fixed- or dynamic-size Eigen type, e.g. as<Eigen::Matrix3d>(j), as<Eigen::Vector3d>(j).
/// Throws std::runtime_error if the size does not match.
template <typename M>
M as(const json& j) {
    Eigen::MatrixXd m;
    if (M::ColsAtCompileTime == 1) {
        m = as_vec(j);
    } else if (M::RowsAtCompileTime == 1) {
        m = as_vec(j).transpose();
    } else {
        m = as_mat(j);
    }
    const bool rows_ok = M::RowsAtCompileTime == Eigen::Dynamic || M::RowsAtCompileTime == m.rows();
    const bool cols_ok = M::ColsAtCompileTime == Eigen::Dynamic || M::ColsAtCompileTime == m.cols();
    if (!rows_ok || !cols_ok)
        throw std::runtime_error("JSON value has shape " + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) +
                                 ", expected " + std::to_string(M::RowsAtCompileTime) + "x" +
                                 std::to_string(M::ColsAtCompileTime) + ": " + j.dump());
    return M(m);
}

/// JSON list -> std::vector of a fixed- or dynamic-size Eigen type, e.g. as_list<Eigen::Matrix3d>(j).
template <typename M>
std::vector<M> as_list(const json& j) {
    if (!j.is_array()) throw std::runtime_error("expected a JSON list, got: " + j.dump());
    std::vector<M> out;
    out.reserve(j.size());
    for (const auto& e : j) out.push_back(as<M>(e));
    return out;
}

/// JSON list of booleans (or 0/1) -> std::vector<bool>.
inline std::vector<bool> as_bools(const json& j) {
    if (!j.is_array()) throw std::runtime_error("expected a JSON list of booleans, got: " + j.dump());
    std::vector<bool> out;
    for (const auto& e : j) out.push_back(e.is_boolean() ? e.get<bool>() : as_double(e) != 0.0);
    return out;
}

/// Eigen -> JSON (vector: flat list; matrix: list of rows).
template <typename Derived>
json to_json(const Eigen::MatrixBase<Derived>& a) {
    json j = json::array();
    if (a.cols() == 1) {
        for (Eigen::Index i = 0; i < a.rows(); ++i) j.push_back(static_cast<double>(a(i, 0)));
        return j;
    }
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        json row = json::array();
        for (Eigen::Index k = 0; k < a.cols(); ++k) row.push_back(static_cast<double>(a(i, k)));
        j.push_back(row);
    }
    return j;
}

// ------------------------------------------------------------------ CSV
struct CsvTable {
    std::vector<std::string> header;  ///< column names from the first line
    Eigen::MatrixXd data;             ///< one row per data line
    /// Index of a column by name (throws if absent).
    Eigen::Index col(const std::string& name) const {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::runtime_error("CSV has no column '" + name + "'");
        return static_cast<Eigen::Index>(it - header.begin());
    }
};

namespace detail {
inline std::vector<std::string> split_csv_line(std::string line, char delim) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> out;
    std::string cell;
    std::istringstream ss(line);
    while (std::getline(ss, cell, delim)) {
        const auto b = cell.find_first_not_of(" \t\"");
        const auto e = cell.find_last_not_of(" \t\"");
        out.push_back(b == std::string::npos ? std::string() : cell.substr(b, e - b + 1));
    }
    if (!line.empty() && line.back() == delim) out.emplace_back();
    return out;
}
}  // namespace detail

/// Read a numeric CSV file whose first line is a header (np.loadtxt(path, delimiter=",", skiprows=1)).
/// Empty lines and lines starting with '#' are skipped. Throws std::runtime_error on a non-numeric cell.
inline CsvTable read_csv(const std::string& path, char delim = ',', bool has_header = true) {
    std::ifstream in = open_input(path);
    CsvTable t;
    std::string line;
    std::vector<std::vector<double>> rows;
    bool first = true;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (first && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF) line.erase(0, 3);  // UTF-8 BOM
        if (line.empty() || line == "\r" || line[0] == '#') continue;
        auto cells = detail::split_csv_line(line, delim);
        if (first && has_header) {
            t.header = cells;
            first = false;
            continue;
        }
        first = false;
        std::vector<double> row;
        for (const auto& c : cells) {
            try {
                std::size_t used = 0;
                row.push_back(std::stod(c, &used));
                if (used != c.size()) throw std::invalid_argument(c);
            } catch (const std::exception&) {
                throw std::runtime_error(path + ":" + std::to_string(lineno) + ": not a number: '" + c + "'");
            }
        }
        if (!rows.empty() && row.size() != rows[0].size())
            throw std::runtime_error(path + ":" + std::to_string(lineno) + ": wrong number of columns");
        rows.push_back(std::move(row));
    }
    const std::size_t c = rows.empty() ? t.header.size() : rows[0].size();
    t.data.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(c));
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t k = 0; k < c; ++k)
            t.data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = rows[i][k];
    return t;
}

/// Write a numeric table as CSV (full double precision) for plotting in Python.
inline void write_csv(const std::string& path, const std::vector<std::string>& header, const Eigen::MatrixXd& data) {
    std::ofstream out(fs_path(path), std::ios::binary);
    if (!out) throw std::runtime_error("cannot write file: " + path);
    for (std::size_t k = 0; k < header.size(); ++k) out << (k ? "," : "") << header[k];
    if (!header.empty()) out << "\n";
    out << std::setprecision(17);
    for (Eigen::Index i = 0; i < data.rows(); ++i) {
        for (Eigen::Index k = 0; k < data.cols(); ++k) out << (k ? "," : "") << data(i, k);
        out << "\n";
    }
}

// ------------------------------------------------------------------ approximate comparison
struct NearResult {
    bool ok = false;
    std::string msg;
};

inline std::string mat_to_string(const Eigen::MatrixXd& m) {
    std::ostringstream s;
    if (m.cols() == 1) {
        s << "[" << m.transpose().format(Eigen::IOFormat(10, Eigen::DontAlignCols, ", ", ", ")) << "]";
    } else {
        s << "\n" << m.format(Eigen::IOFormat(10, 0, ", ", "\n", "    [", "]"));
    }
    return s.str();
}

/// Compare two matrices/vectors entry by entry: ok if same shape and max |got - expected| <= tol.
/// Two vectors with the same length are compared even if one is a row and the other a column.
template <typename A, typename B>
NearResult compare_near(const Eigen::MatrixBase<A>& got_in, const Eigen::MatrixBase<B>& exp_in, double tol) {
    Eigen::MatrixXd got = got_in.template cast<double>();
    Eigen::MatrixXd exp = exp_in.template cast<double>();
    const bool got_vec = got.rows() == 1 || got.cols() == 1, exp_vec = exp.rows() == 1 || exp.cols() == 1;
    if (got_vec && exp_vec && got.size() == exp.size()) {
        got = Eigen::Map<Eigen::VectorXd>(got.data(), got.size()).eval();
        exp = Eigen::Map<Eigen::VectorXd>(exp.data(), exp.size()).eval();
    }
    NearResult r;
    std::ostringstream s;
    if (got.rows() != exp.rows() || got.cols() != exp.cols()) {
        s << "shape mismatch: got " << got.rows() << "x" << got.cols() << ", expected " << exp.rows() << "x"
          << exp.cols();
        r.msg = s.str();
        return r;
    }
    if (got.size() == 0) {
        r.ok = true;
        return r;
    }
    if (!got.allFinite()) {
        s << "result contains NaN or inf:" << mat_to_string(got);
        r.msg = s.str();
        return r;
    }
    Eigen::Index i = 0, k = 0;
    const double err = (got - exp).cwiseAbs().maxCoeff(&i, &k);
    r.ok = err <= tol;
    s << std::setprecision(3) << "max |got - expected| = " << err << " at (" << i << "," << k << "), tolerance "
      << tol << "\n  got:      " << mat_to_string(got) << "\n  expected: " << mat_to_string(exp);
    r.msg = s.str();
    return r;
}

inline NearResult compare_near(double got, double exp, double tol) {
    NearResult r;
    r.ok = std::isfinite(got) && std::abs(got - exp) <= tol;
    std::ostringstream s;
    s << std::setprecision(12) << "got " << got << ", expected " << exp << " (|difference| = " << std::abs(got - exp)
      << ", tolerance " << tol << ")";
    r.msg = s.str();
    return r;
}

/// Tolerance used by pytest.approx(expected) with its defaults: max(rel * |expected|, abs).
inline double approx_tol(double expected, double rel = 1e-6, double abs_tol = 1e-12) {
    return std::max(rel * std::abs(expected), abs_tol);
}

}  // namespace sbeg

// doctest macros (include <doctest/doctest.h> before using them)
#define CHECK_MAT_NEAR(got, expected, tol)                                                              \
    do {                                                                                                \
        const ::sbeg::NearResult sbeg_near_ = ::sbeg::compare_near((got), (expected), (tol));                   \
        CHECK_MESSAGE(sbeg_near_.ok, "CHECK_MAT_NEAR(" #got ", " #expected ")\n  " << sbeg_near_.msg);  \
    } while (false)

#define REQUIRE_MAT_NEAR(got, expected, tol)                                                              \
    do {                                                                                                  \
        const ::sbeg::NearResult sbeg_near_ = ::sbeg::compare_near((got), (expected), (tol));                     \
        REQUIRE_MESSAGE(sbeg_near_.ok, "REQUIRE_MAT_NEAR(" #got ", " #expected ")\n  " << sbeg_near_.msg); \
    } while (false)

#define CHECK_SCALAR_NEAR(got, expected, tol)                                                                    \
    do {                                                                                                         \
        const ::sbeg::NearResult sbeg_near_ =                                                                    \
            ::sbeg::compare_near(static_cast<double>(got), static_cast<double>(expected), (tol));                        \
        CHECK_MESSAGE(sbeg_near_.ok, "CHECK_SCALAR_NEAR(" #got ", " #expected ")\n  " << sbeg_near_.msg);        \
    } while (false)

#define CHECK_APPROX(got, expected)                                                                           \
    do {                                                                                                      \
        const double sbeg_exp_ = static_cast<double>(expected);                                               \
        const ::sbeg::NearResult sbeg_near_ =                                                                 \
            ::sbeg::compare_near(static_cast<double>(got), sbeg_exp_, ::sbeg::approx_tol(sbeg_exp_));          \
        CHECK_MESSAGE(sbeg_near_.ok, "CHECK_APPROX(" #got ", " #expected ")\n  " << sbeg_near_.msg);           \
    } while (false)

#endif  // SBEG_LABIO_HPP
