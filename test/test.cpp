#include "fgt.hpp"

int main() {
  // Read ndim, nx
  int ndim, nx;
  std::cout << "enter ndim, nx\n";
  std::cin >> ndim >> nx;
  std::cout << "ndim= " << ndim << ", nx= " << nx << "\n";

  int ny = nx/2;
  int nz = nx/2;
  FGT<double,4>::test(ndim, nx, ny, nz);

  return 0;
}
