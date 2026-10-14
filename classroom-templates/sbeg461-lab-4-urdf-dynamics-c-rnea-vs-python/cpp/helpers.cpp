// SBEG 461 Lab 4 -- provided helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   skew, rpy_to_R, rp_to_trans, trans_inv, adjoint, ad, matrix_exp6   small SE(3) toolkit
//   read_urdf_chain                                                     URDF reading (a minimal XML reader is included)
//   load_states, export_for_cpp, benchmark                              data files, Step 3 files and timing
//
// These are the provided helpers at the top of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>
#include <sbeg/mrcpp.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Small SE(3) toolkit
// ---------------------------------------------------------------------------

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d W;
    W << 0.0, -w(2), w(1),
         w(2), 0.0, -w(0),
         -w(1), w(0), 0.0;
    return W;
}

Eigen::Matrix3d rpy_to_R(const Eigen::Vector3d& rpy) {
    const double cr = std::cos(rpy(0)), sr = std::sin(rpy(0));
    const double cp = std::cos(rpy(1)), sp = std::sin(rpy(1));
    const double cy = std::cos(rpy(2)), sy = std::sin(rpy(2));
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr;
    Ry << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;
    Rz << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0;
    return Rz * Ry * Rx;
}

Eigen::Matrix4d rp_to_trans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.block<3, 1>(0, 3) = p;
    return T;
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d Rt = T.topLeftCorner<3, 3>().transpose();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    return rp_to_trans(Rt, -Rt * p);
}

Matrix6d adjoint(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    Matrix6d A = Matrix6d::Zero();
    A.topLeftCorner<3, 3>() = R;
    A.bottomLeftCorner<3, 3>() = skew(p) * R;
    A.bottomRightCorner<3, 3>() = R;
    return A;
}

Matrix6d ad(const Vector6d& V) {
    const Eigen::Matrix3d W = skew(V.head<3>());
    Matrix6d out = Matrix6d::Zero();
    out.topLeftCorner<3, 3>() = W;
    out.bottomLeftCorner<3, 3>() = skew(V.tail<3>());
    out.bottomRightCorner<3, 3>() = W;
    return out;
}

Eigen::Matrix4d matrix_exp6(const Vector6d& S, double theta) {
    const Eigen::Vector3d w = S.head<3>(), v = S.tail<3>();
    if (w.norm() < 1e-12) return rp_to_trans(Eigen::Matrix3d::Identity(), v * theta);
    const Eigen::Matrix3d W = skew(w), W2 = W * W, I3 = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R = I3 + std::sin(theta) * W + (1.0 - std::cos(theta)) * W2;
    const Eigen::Vector3d Gv = (I3 * theta + (1.0 - std::cos(theta)) * W + (theta - std::sin(theta)) * W2) * v;
    return rp_to_trans(R, Gv);
}

// ---------------------------------------------------------------------------
// Minimal XML reader (enough for URDF files: elements, attributes, comments; no DTD, no entities in names)
// ---------------------------------------------------------------------------
namespace {

struct XmlElement {
    std::string tag;
    std::map<std::string, std::string> attrs;
    std::vector<XmlElement> children;
    /// First direct child with this tag, or nullptr (Python: element.find(tag)).
    const XmlElement* find(const std::string& t) const {
        for (const auto& c : children)
            if (c.tag == t) return &c;
        return nullptr;
    }
    /// Attribute value, or `fallback` if absent (Python: element.get(name, fallback)).
    std::string get(const std::string& name, const std::string& fallback = "") const {
        const auto it = attrs.find(name);
        return it == attrs.end() ? fallback : it->second;
    }
    bool has(const std::string& name) const { return attrs.count(name) > 0; }
};

std::string xml_unescape(const std::string& s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '&') {
            const std::size_t e = s.find(';', i);
            if (e != std::string::npos) {
                const std::string ent = s.substr(i + 1, e - i - 1);
                const std::map<std::string, char> known = {
                    {"lt", '<'}, {"gt", '>'}, {"amp", '&'}, {"quot", '"'}, {"apos", '\''}};
                const auto it = known.find(ent);
                if (it != known.end()) {
                    out += it->second;
                    i = e;
                    continue;
                }
            }
        }
        out += s[i];
    }
    return out;
}

/// Parse the text of an XML document and return its root element.
XmlElement parse_xml(const std::string& text, const std::string& path) {
    std::vector<XmlElement> stack;  // open elements
    XmlElement root;
    bool have_root = false;
    std::size_t i = 0;
    auto fail = [&](const std::string& what) {
        throw std::runtime_error("XML error in " + path + " near character " + std::to_string(i) + ": " + what);
    };
    while (true) {
        const std::size_t lt = text.find('<', i);
        if (lt == std::string::npos) break;
        i = lt;
        if (text.compare(i, 4, "<!--") == 0) {  // comment
            const std::size_t e = text.find("-->", i + 4);
            if (e == std::string::npos) fail("unterminated comment");
            i = e + 3;
            continue;
        }
        if (text.compare(i, 2, "<?") == 0 || text.compare(i, 2, "<!") == 0) {  // declaration / DOCTYPE
            const std::size_t e = text.find('>', i);
            if (e == std::string::npos) fail("unterminated declaration");
            i = e + 1;
            continue;
        }
        // find the end of the tag, skipping '>' inside quoted attribute values
        std::size_t e = i + 1;
        char quote = 0;
        for (; e < text.size(); ++e) {
            const char ch = text[e];
            if (quote) {
                if (ch == quote) quote = 0;
            } else if (ch == '"' || ch == '\'') {
                quote = ch;
            } else if (ch == '>') {
                break;
            }
        }
        if (e >= text.size()) fail("unterminated tag");
        std::string body = text.substr(i + 1, e - i - 1);
        i = e + 1;
        if (!body.empty() && body[0] == '/') {  // closing tag
            if (stack.empty()) fail("unexpected closing tag");
            XmlElement done = std::move(stack.back());
            stack.pop_back();
            if (stack.empty()) {
                root = std::move(done);
                have_root = true;
            } else {
                stack.back().children.push_back(std::move(done));
            }
            continue;
        }
        const bool self_closing = !body.empty() && body.back() == '/';
        if (self_closing) body.pop_back();
        XmlElement el;
        std::size_t k = 0;
        while (k < body.size() && !std::isspace(static_cast<unsigned char>(body[k]))) el.tag += body[k++];
        while (k < body.size()) {  // attributes  name = "value"
            while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
            if (k >= body.size()) break;
            std::string name;
            while (k < body.size() && body[k] != '=' && !std::isspace(static_cast<unsigned char>(body[k])))
                name += body[k++];
            while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
            if (k >= body.size() || body[k] != '=') fail("attribute '" + name + "' has no value");
            ++k;
            while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
            if (k >= body.size() || (body[k] != '"' && body[k] != '\'')) fail("attribute value not quoted");
            const char q = body[k++];
            const std::size_t end = body.find(q, k);
            if (end == std::string::npos) fail("unterminated attribute value");
            el.attrs[name] = xml_unescape(body.substr(k, end - k));
            k = end + 1;
        }
        if (self_closing) {
            if (stack.empty()) {
                root = std::move(el);
                have_root = true;
            } else {
                stack.back().children.push_back(std::move(el));
            }
        } else {
            stack.push_back(std::move(el));
        }
    }
    if (!stack.empty() || !have_root) throw std::runtime_error("XML error in " + path + ": incomplete document");
    return root;
}

Eigen::Vector3d parse_vec3(const std::string& s, const std::string& what) {
    std::istringstream in(s);
    in.imbue(std::locale::classic());
    Eigen::Vector3d v;
    for (int k = 0; k < 3; ++k)
        if (!(in >> v(k))) throw std::runtime_error("URDF: cannot read three numbers from " + what + "=\"" + s + "\"");
    return v;
}

double parse_double(const std::string& s) {
    std::istringstream in(s);
    in.imbue(std::locale::classic());
    double x = 0.0;
    if (!(in >> x)) throw std::runtime_error("URDF: not a number: \"" + s + "\"");
    return x;
}

}  // namespace

// ---------------------------------------------------------------------------
// URDF
// ---------------------------------------------------------------------------

std::vector<UrdfJoint> read_urdf_chain(const std::string& path) {
    std::ifstream in = sbeg::open_input(path);
    std::stringstream buf;
    buf << in.rdbuf();
    const XmlElement root = parse_xml(buf.str(), path);

    struct LinkInertia {
        double mass = 0.0;
        Eigen::Matrix4d T_com = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d I = Eigen::Matrix3d::Zero();
    };
    std::vector<std::string> link_order;  // document order (the Python dict keeps it)
    std::map<std::string, LinkInertia> links;
    for (const XmlElement& link : root.children) {
        if (link.tag != "link") continue;
        const std::string name = link.get("name");
        if (!links.count(name)) link_order.push_back(name);
        LinkInertia L;
        if (const XmlElement* inertial = link.find("inertial")) {
            Eigen::Vector3d xyz = Eigen::Vector3d::Zero(), rpy = Eigen::Vector3d::Zero();
            if (const XmlElement* org = inertial->find("origin")) {
                xyz = parse_vec3(org->get("xyz", "0 0 0"), "xyz");
                rpy = parse_vec3(org->get("rpy", "0 0 0"), "rpy");
            }
            const XmlElement* mass = inertial->find("mass");
            const XmlElement* inertia = inertial->find("inertia");
            if (!mass || !inertia) throw std::runtime_error("URDF: link '" + name + "' has no <mass>/<inertia>");
            L.mass = parse_double(mass->get("value"));
            auto a = [&](const char* key) { return parse_double(inertia->get(key, "0")); };
            L.I << a("ixx"), a("ixy"), a("ixz"),
                   a("ixy"), a("iyy"), a("iyz"),
                   a("ixz"), a("iyz"), a("izz");
            L.T_com = rp_to_trans(rpy_to_R(rpy), xyz);
        }
        links[name] = L;
    }

    std::map<std::string, const XmlElement*> joints;  // parent link -> joint (last one wins, as the Python dict)
    std::set<std::string> children;
    std::size_t n_joints = 0;
    for (const XmlElement& j : root.children) {
        if (j.tag != "joint") continue;
        const XmlElement* parent = j.find("parent");
        const XmlElement* child = j.find("child");
        if (!parent || !child) throw std::runtime_error("URDF: joint '" + j.get("name") + "' has no <parent>/<child>");
        joints[parent->get("link")] = &j;
        children.insert(child->get("link"));
        ++n_joints;
    }
    std::string parent;
    for (const std::string& name : link_order)
        if (!children.count(name)) {
            parent = name;
            break;
        }
    if (parent.empty()) throw std::runtime_error("URDF: no base link found in " + path);

    std::vector<UrdfJoint> chain;
    while (joints.count(parent) && chain.size() <= n_joints) {  // size guard: no endless loop on a cycle
        const XmlElement& j = *joints[parent];
        UrdfJoint J;
        J.name = j.get("name");
        J.type = j.get("type");
        Eigen::Vector3d xyz = Eigen::Vector3d::Zero(), rpy = Eigen::Vector3d::Zero();
        if (const XmlElement* org = j.find("origin")) {
            xyz = parse_vec3(org->get("xyz", "0 0 0"), "xyz");
            rpy = parse_vec3(org->get("rpy", "0 0 0"), "rpy");
        }
        J.T_origin = rp_to_trans(rpy_to_R(rpy), xyz);
        if (J.type != "fixed") {
            const XmlElement* ax = j.find("axis");
            J.axis = ax ? parse_vec3(ax->get("xyz"), "axis xyz") : Eigen::Vector3d(1.0, 0.0, 0.0);
            J.axis /= J.axis.norm();
        }
        J.child = j.find("child")->get("link");
        const auto it = links.find(J.child);
        if (it == links.end()) throw std::runtime_error("URDF: joint '" + J.name + "' has an unknown child link");
        J.mass = it->second.mass;
        J.T_com = it->second.T_com;
        J.I = it->second.I;
        chain.push_back(J);
        parent = J.child;
    }
    return chain;
}

// ---------------------------------------------------------------------------
// Data files, Step 3 files and timing
// ---------------------------------------------------------------------------

Eigen::MatrixXd load_states(const std::string& csv_path) { return sbeg::read_csv(csv_path).data; }

namespace {
template <typename Derived>
void write_rows(std::ostream& f, const Eigen::MatrixBase<Derived>& A) {  // = np.savetxt(f, A, fmt="%.17g")
    for (Eigen::Index r = 0; r < A.rows(); ++r) {
        for (Eigen::Index c = 0; c < A.cols(); ++c) f << (c ? " " : "") << A(r, c);
        f << "\n";
    }
}
}  // namespace

void export_for_cpp(const std::string& model_path, const std::string& states_path, const TransformList& Mlist,
                    const InertiaList& Glist, const Eigen::MatrixXd& Slist, const Eigen::MatrixXd& states) {
    std::ofstream f(sbeg::fs_path(model_path));
    if (!f) throw std::runtime_error("cannot write " + model_path);
    f.imbue(std::locale::classic());
    f << std::setprecision(17) << Glist.size() << "\n";
    for (const Eigen::Matrix4d& M : Mlist) write_rows(f, M);
    for (const Matrix6d& G : Glist) write_rows(f, G);
    write_rows(f, Slist.transpose());
    std::ofstream s(sbeg::fs_path(states_path));
    if (!s) throw std::runtime_error("cannot write " + states_path);
    s.imbue(std::locale::classic());
    s << std::setprecision(17);
    write_rows(s, states);
}

std::map<std::string, double> benchmark(const std::string& urdf_path, const std::string& csv_path, int n_states) {
    const MRModel m = urdf_to_mr(urdf_path);
    const Eigen::MatrixXd all = load_states(csv_path);
    const Eigen::MatrixXd X = all.topRows(std::min<Eigen::Index>(n_states, all.rows()));
    const Eigen::Vector3d g(0.0, 0.0, -9.81);
    const Vector6d Ftip = Vector6d::Zero();
    const auto n = static_cast<std::size_t>(X.rows());
    volatile double sink = 0.0;  // keeps the optimiser from removing the timed calls
    auto row = [&](std::size_t k, Eigen::Index part) {
        return Eigen::VectorXd(X.row(static_cast<Eigen::Index>(k)).segment(6 * part, 6).transpose());
    };
    std::map<std::string, double> out;
    out["cpp_own_rnea"] = 1e6 * time_per_call([&](std::size_t k) {
        sink = sink + rnea(row(k, 0), row(k, 1), row(k, 2), g, Ftip, m.Mlist, m.Glist, m.Slist)(0);
    }, n);
    out["mrcpp_inverse_dynamics"] = 1e6 * time_per_call([&](std::size_t k) {
        sink = sink + mr::inverse_dynamics(row(k, 0), row(k, 1), row(k, 2), g, Ftip, m.Mlist, m.Glist, m.Slist)(0);
    }, n);
    return out;
}

}  // namespace lab
