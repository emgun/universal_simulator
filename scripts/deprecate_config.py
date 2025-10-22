#!/usr/bin/env python3
"""
Deprecate configuration files with proper notice.

Usage:
    python scripts/deprecate_config.py configs/old_config.yaml \
        --reason "Superseded by train_burgers_golden.yaml" \
        --replacement train_burgers_golden.yaml

    python scripts/deprecate_config.py configs/old_config.yaml \
        --reason "Experimental config, not validated" \
        --move-to-deprecated
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def add_deprecation_notice(
    config_path: Path,
    reason: str,
    replacement: str | None = None,
    date: str | None = None
) -> None:
    """Add deprecation notice to top of config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    date = date or datetime.now().strftime("%Y-%m-%d")

    # Read existing content
    content = config_path.read_text()

    # Build deprecation notice
    notice_lines = [
        f"# DEPRECATED: {date}",
    ]

    if replacement:
        notice_lines.append(f"# This config has been superseded by {replacement}")
        notice_lines.append(f"# Use {replacement} instead for all new work.")
    else:
        notice_lines.append("# This config is deprecated and should not be used.")

    notice_lines.append(f"# Reason: {reason}")
    notice_lines.append("#")

    notice = "\n".join(notice_lines) + "\n"

    # Check if already deprecated
    if "DEPRECATED:" in content[:200]:
        print(f"⚠️  {config_path.name} is already marked as deprecated")
        response = input("Overwrite existing deprecation notice? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return

        # Remove old deprecation notice (first block of comments)
        lines = content.split('\n')
        new_lines = []
        skip_deprecation = False
        found_deprecation = False

        for line in lines:
            if "DEPRECATED:" in line:
                found_deprecation = True
                skip_deprecation = True
                continue
            if skip_deprecation:
                if line.startswith('#'):
                    continue
                else:
                    skip_deprecation = False

            if not (skip_deprecation or (found_deprecation and not line.strip())):
                new_lines.append(line)

        content = '\n'.join(new_lines).lstrip('\n')

    # Add new notice
    new_content = notice + content

    # Write back
    config_path.write_text(new_content)
    print(f"✅ Added deprecation notice to {config_path.name}")


def move_to_deprecated_dir(config_path: Path, configs_dir: Path) -> None:
    """Move config to deprecated/ subdirectory."""
    deprecated_dir = configs_dir / "deprecated"
    deprecated_dir.mkdir(exist_ok=True)

    dest = deprecated_dir / config_path.name

    if dest.exists():
        print(f"⚠️  {dest} already exists")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping move...")
            return

    shutil.move(str(config_path), str(dest))
    print(f"✅ Moved {config_path.name} → {dest.relative_to(configs_dir.parent)}")


def update_readme(
    configs_dir: Path,
    config_name: str,
    date: str,
    reason: str,
    replacement: str | None = None
) -> None:
    """Update configs/README.md with deprecation entry."""
    readme_path = configs_dir / "README.md"

    if not readme_path.exists():
        print(f"⚠️  {readme_path} not found, skipping README update")
        return

    content = readme_path.read_text()

    # Find the deprecated configs table
    if "## Deprecated Configs" not in content:
        print("⚠️  No 'Deprecated Configs' section found in README")
        return

    # Build new table row
    replacement_text = f"`{replacement}`" if replacement else "N/A"
    new_row = f"| `{config_name}` | {date} | {reason} | {replacement_text} |"

    # Check if already in table
    if config_name in content:
        print(f"⚠️  {config_name} already in README deprecation table")
        response = input("Update entry? (y/n): ")
        if response.lower() != 'y':
            print("Skipping README update...")
            return

        # Replace existing row
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if f"| `{config_name}`" in line:
                new_lines.append(new_row)
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    else:
        # Add new row after table header
        lines = content.split('\n')
        new_lines = []
        inserted = False

        for i, line in enumerate(lines):
            new_lines.append(line)
            # Look for table header in Deprecated Configs section
            if not inserted and "## Deprecated Configs" in line:
                # Find the table separator (|---|---|)
                for j in range(i, min(i + 10, len(lines))):
                    if lines[j].startswith('|---'):
                        # Insert after separator
                        new_lines.append(new_row)
                        inserted = True
                        break

        if not inserted:
            print("⚠️  Could not find table location in README")
            return

        content = '\n'.join(new_lines)

    readme_path.write_text(content)
    print(f"✅ Updated {readme_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Deprecate a configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deprecate config with replacement
  python scripts/deprecate_config.py configs/old_config.yaml \\
      --reason "Superseded by golden config" \\
      --replacement train_burgers_golden.yaml

  # Deprecate and move to deprecated/
  python scripts/deprecate_config.py configs/old_config.yaml \\
      --reason "Experimental, not validated" \\
      --move-to-deprecated

  # Dry-run
  python scripts/deprecate_config.py configs/old_config.yaml \\
      --reason "Testing" \\
      --dry-run
        """
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to config file to deprecate"
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Reason for deprecation"
    )
    parser.add_argument(
        "--replacement",
        help="Replacement config to use instead"
    )
    parser.add_argument(
        "--date",
        help="Deprecation date (default: today)"
    )
    parser.add_argument(
        "--move-to-deprecated",
        action="store_true",
        help="Move config to configs/deprecated/ directory"
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        default=True,
        help="Update configs/README.md (default: true)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    configs_dir = config_path.parent

    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    if not config_path.suffix == '.yaml':
        parser.error(f"Not a YAML file: {config_path}")

    date = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Deprecating: {config_path.name}")
    print(f"Date: {date}")
    print(f"Reason: {args.reason}")
    if args.replacement:
        print(f"Replacement: {args.replacement}")
    if args.move_to_deprecated:
        print(f"Move to: configs/deprecated/{config_path.name}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] Would perform the following actions:")
        print(f"  1. Add deprecation notice to {config_path.name}")
        if args.update_readme:
            print(f"  2. Update configs/README.md")
        if args.move_to_deprecated:
            print(f"  3. Move to configs/deprecated/")
        print("\nRun without --dry-run to execute")
        return

    # Perform deprecation
    try:
        # 1. Add deprecation notice
        add_deprecation_notice(
            config_path,
            args.reason,
            args.replacement,
            date
        )

        # 2. Update README
        if args.update_readme:
            update_readme(
                configs_dir,
                config_path.name,
                date,
                args.reason,
                args.replacement
            )

        # 3. Move to deprecated/ (if requested)
        if args.move_to_deprecated:
            move_to_deprecated_dir(config_path, configs_dir)

        print(f"\n✅ Successfully deprecated {config_path.name}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
