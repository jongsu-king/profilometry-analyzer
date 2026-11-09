# Security

Profilometry Analyzer reads the scan files you choose and writes results to a folder you
choose. The only network access is the optional release check (one anonymous request to
the GitHub API that sends only the program version, from the desktop application at
start-up; never from the engine or from batch runs). Disable it with *Help > Check for
updates at startup* or the environment variable `PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK=1`.

The installers are not code-signed; verify a download against `SHA256SUMS.txt` on the
release page before running it.

To report a vulnerability, open a private security advisory on GitHub
(*Security > Report a vulnerability*) rather than a public issue.
