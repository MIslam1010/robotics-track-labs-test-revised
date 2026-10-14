// SBEG 361 Lab 5 -- complete helpers (C++ track). Given code: do not edit; use these functions freely.
//
//   vec_to_so3, vec_to_se3, trans_inv, adjoint, rot_axis      helpers from Labs 1-3
//   read_urdf_joints, urdf_chain                              URDF reading (a minimal XML reader is included below)
//   load_joint_log, load_offsets                              hardware log and calibration file
//
// These are the "(complete helper)" functions of starter.py, translated to C++.
#include "lab.hpp"

#include <sbeg/labio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>

namespace lab {

// ---------------------------------------------------------------------------
// Helpers from Labs 1-3
// ---------------------------------------------------------------------------

Eigen::Matrix3d vec_to_so3(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d m;
    m << 0.0, -omega(2), omega(1),
         omega(2), 0.0, -omega(0),
         -omega(1), omega(0), 0.0;
    return m;
}

Eigen::Matrix4d vec_to_se3(const sbeg::Vector6d& V) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
    m.topLeftCorner<3, 3>() = vec_to_so3(V.head<3>());
    m.block<3, 1>(0, 3) = V.tail<3>();
    return m;
}

Eigen::Matrix4d trans_inv(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    Ti.topLeftCorner<3, 3>() = R.transpose();
    Ti.block<3, 1>(0, 3) = -R.transpose() * p;
    return Ti;
}

sbeg::Matrix6d adjoint(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d p = T.block<3, 1>(0, 3);
    sbeg::Matrix6d A = sbeg::Matrix6d::Zero();
    A.topLeftCorner<3, 3>() = R;
    A.bottomRightCorner<3, 3>() = R;
    A.bottomLeftCorner<3, 3>() = vec_to_so3(p) * R;
    return A;
}

Eigen::Matrix3d rot_axis(const std::string& axis, double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    Eigen::Matrix3d R;
    if (axis == "x") {
        R << 1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c;
    } else if (axis == "y") {
        R << c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c;
    } else {
        R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
    }
    return R;
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

std::vector<UrdfJoint> read_urdf_joints(const std::string& path) {
    std::ifstream in = sbeg::open_input(path);
    std::stringstream buf;
    buf << in.rdbuf();
    const XmlElement root = parse_xml(buf.str(), path);
    std::vector<UrdfJoint> joints;
    for (const XmlElement& j : root.children) {  // direct children of <robot> only (as root.findall("joint"))
        if (j.tag != "joint") continue;
        UrdfJoint J;
        J.name = j.get("name");
        J.type = j.get("type");
        const XmlElement* parent = j.find("parent");
        const XmlElement* child = j.find("child");
        if (!parent || !child) throw std::runtime_error("URDF: joint '" + J.name + "' has no <parent>/<child>");
        J.parent = parent->get("link");
        J.child = child->get("link");
        if (const XmlElement* o = j.find("origin")) {
            J.xyz = parse_vec3(o->get("xyz", "0 0 0"), "xyz");
            J.rpy = parse_vec3(o->get("rpy", "0 0 0"), "rpy");
        }
        if (const XmlElement* a = j.find("axis")) J.axis = parse_vec3(a->get("xyz"), "axis xyz");
        if (const XmlElement* lim = j.find("limit"); lim && lim->has("lower"))
            J.limit = std::make_pair(parse_double(lim->get("lower")), parse_double(lim->get("upper")));
        joints.push_back(J);
    }
    return joints;
}

std::vector<UrdfJoint> urdf_chain(const std::vector<UrdfJoint>& joints, const std::string& tip_link) {
    std::map<std::string, const UrdfJoint*> by_child;
    for (const auto& j : joints) by_child[j.child] = &j;  // last one wins, as in the Python dict
    std::vector<UrdfJoint> chain;
    std::string link = tip_link;
    while (by_child.count(link) && chain.size() <= joints.size()) {  // size guard: no endless loop on a cycle
        const UrdfJoint* j = by_child[link];
        chain.push_back(*j);
        link = j->parent;
    }
    if (chain.empty()) throw std::invalid_argument("no joint leads to link '" + tip_link + "'");
    std::reverse(chain.begin(), chain.end());
    return chain;
}

// ---------------------------------------------------------------------------
// Hardware log and calibration file
// ---------------------------------------------------------------------------

JointLog load_joint_log(const std::string& path) {
    const sbeg::CsvTable tab = sbeg::read_csv(path);  // header line + numeric rows, empty lines skipped
    int n = 0;
    for (const auto& h : tab.header)
        if (h.rfind("cmd_", 0) == 0) ++n;
    JointLog log;
    const Eigen::Index N = tab.data.rows();
    log.t = tab.data.col(tab.col("t"));
    log.cmd.resize(N, n);
    log.meas.resize(N, n);
    for (int i = 0; i < n; ++i) {
        log.cmd.col(i) = tab.data.col(tab.col("cmd_" + std::to_string(i + 1)));
        log.meas.col(i) = tab.data.col(tab.col("meas_" + std::to_string(i + 1)));
    }
    if (std::find(tab.header.begin(), tab.header.end(), "stop") != tab.header.end())
        log.stop = tab.data.col(tab.col("stop"));
    return log;
}

Eigen::VectorXd load_offsets(const std::string& path) {
    const sbeg::json j = sbeg::load_json(path);
    if (!j.contains("servo_zero_offset_deg")) throw std::runtime_error(path + ": no 'servo_zero_offset_deg'");
    return sbeg::as_vec(j["servo_zero_offset_deg"]);
}

}  // namespace lab
