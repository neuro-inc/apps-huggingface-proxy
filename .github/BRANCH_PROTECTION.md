# Branch Protection Configuration

This document describes the required branch protection rules for the `master` branch.

## GitHub Repository Settings

Navigate to: **Settings → Branches → Branch protection rules → Add rule**

### Master Branch Protection Rules

Branch name pattern: `master`

#### Required Settings

**Protect matching branches:**
- ✅ Require a pull request before merging
  - ✅ Require approvals: **1**
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners (if CODEOWNERS file exists)

- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - **Required status checks:**
    - `Lint and Test`
    - `Build Docker Image`

- ✅ Require conversation resolution before merging

- ✅ Do not allow bypassing the above settings

#### Recommended Settings

- ✅ Require linear history (optional, for cleaner git history)
- ✅ Include administrators (enforce rules for all users)

## Dependabot Security Alerts

Navigate to: **Settings → Code security and analysis**

Enable the following:
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates

The `.github/dependabot.yml` file configures automated dependency updates.

## Branch Rename (if needed)

If the current default branch is `main`, rename it to `master`:

```bash
# Locally rename branch
git branch -m main master

# Push the new branch
git push -u origin master

# Update GitHub default branch in Settings → Branches → Default branch

# Delete old main branch from remote
git push origin --delete main
```

## CI/CD Configuration

The CI workflow (`.github/workflows/ci.yml`) is configured to:
- Run on all PRs targeting `master`
- Run on all pushes to `master`
- Run on version tags (`v*`)
- Push Docker images only on `master` branch or tags

## Testing Branch Protection

To verify branch protection is working:

1. Create a test branch: `git checkout -b test/branch-protection`
2. Make a change and push: `git push -u origin test/branch-protection`
3. Open a PR to `master` on GitHub
4. Verify you cannot merge until:
   - CI checks pass (Lint and Test, Build Docker Image)
   - At least 1 approval is received
