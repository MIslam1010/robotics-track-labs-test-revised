// SBEG 461 Lab 4 -- model file reader (PROVIDED, do not edit).
#include <fstream>
#include <stdexcept>

#include "rnea.hpp"

Model load_model(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open model file " + path);
  Model m;
  f >> m.n;
  m.Mlist.resize(static_cast<std::size_t>(m.n) + 1);
  m.Glist.resize(static_cast<std::size_t>(m.n));
  m.Slist.resize(static_cast<std::size_t>(m.n));
  for (auto& M : m.Mlist)
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) f >> M(r, c);
  for (auto& G : m.Glist)
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c) f >> G(r, c);
  for (auto& S : m.Slist)
    for (int r = 0; r < 6; ++r) f >> S(r);
  if (!f) throw std::runtime_error("model file is truncated: " + path);
  return m;
}
