# Security policy

## Supported versions

RoboSmith is pre-1.0. Security fixes target the latest release and `main`; older snapshots may not receive
backports.

## Reporting a vulnerability

Use GitHub **Security → Report a vulnerability** to submit a private report. Do not include exploit details,
credentials, or sensitive logs in a public Issue.

Relevant reports include:

- code execution through a crafted asset, URDF, mesh, or scenario file;
- path traversal outside the asset library or output directory;
- Docker images or launch scripts that expose host devices, sockets, or credentials beyond what they document;
- secrets or private paths leaking into generated datasets, videos, or logs.

Include the affected commit, OS, Python version, image tag if any, reproduction steps, and a minimal sanitized
proof. Maintainers will acknowledge the report as soon as practical and coordinate disclosure after a fix is
available.

## Trust boundary

Scenario files are Python modules and run with your privileges; load only scenarios you trust. The e2e launcher
`scripts/setup/run_scenario.sh` mounts the Docker socket into the container, so the container can control the host
Docker daemon. Run it only on machines where that is acceptable.
