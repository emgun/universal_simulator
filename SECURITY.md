# Security Policy

UPS is research software. Treat the repository as executable scientific code:
configs, launch scripts, notebooks, model checkpoints, and data manifests can all
affect local machines or remote compute bills.

## Supported Versions

Security review focuses on the default branch and current release tags, if any.
Historical research branches may contain obsolete launch scripts or experiment
state and should not be treated as supported deployment surfaces.

## Reporting A Vulnerability

Use GitHub private vulnerability reporting for this repository when available.
If a private report path is not available, open a GitHub issue requesting a
private security contact and include only a minimal description until a private
channel is established.

Do not publish working exploit details, live credentials, private data paths, or
cloud instance access details in public issues or pull requests.

## Secret Handling

- Never commit `.env`, cloud credentials, W&B keys, SSH keys, provider tokens, or
  copied launch commands containing literal secrets.
- Prefer environment variables or local ignored files for credentials.
- Dry-run output from launch tooling should redact secrets before printing.
- If a credential is committed or printed publicly, consider it exposed and
  rotate it before continuing.

## Remote Compute Safety

Remote launch helpers should fail closed when required data, credentials,
tracking, or artifact-publication settings are missing. Pull requests that
change launch behavior should explain how they avoid accidental paid runs,
orphaned instances, and untracked benchmark artifacts.
