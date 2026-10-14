# robotics-track-labs-test-revised

Automatic test bench for the student lab repositories of SBEG 361, SBEG 461 and SBEG 46X (Python and C++ tracks).
Every push to `main` builds and runs every lab on Ubuntu, Windows and macOS: the untouched starter must build and fail
cleanly, and the reference solution must pass. The reference solutions are stored **encrypted**
(`solutions.tar.gz.enc`) and are decrypted only inside the test run with a repository secret; the logs show scores
only. The student versions of the labs (no solutions) are in `sbeg361-labs/`, `sbeg461-labs/` and `sbeg46x-labs/`.
