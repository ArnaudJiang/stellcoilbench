# Forking StellCoilBench

This guide explains how to fork StellCoilBench and work with Git LFS, which is used for large files in `submissions/` and `cases/done/` (VTK meshes, plots, JSON).

## How LFS Fork Isolation Works

Submissions and autopilot results in `cases/done/` use **Git LFS** with a dedicated storage repo (`stellcoilbench-lfs`) that only the upstream owner has write access to. This ensures:

- **Forks cannot push to upstream's LFS** — attempts fail with 403 Forbidden.
- **Forks can pull** — LFS objects are readable by everyone.
- **Upstream quota is protected** — fork contributors cannot run up the owner's LFS bill.

## Clone Options

### Full clone (including submissions and cases/done, ~17 GB+)

```bash
git clone https://github.com/akaptano/stellcoilbench.git
cd stellcoilbench
git lfs install
git lfs pull
```

### Code-only clone (skip LFS, small)

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/akaptano/stellcoilbench.git
cd stellcoilbench
```

Use this when you only need source code, tests, or cases — not submission data.

### Verifying LFS (post-migration)

After cloning, confirm LFS objects are present: `git lfs fsck` (no missing/deleted) and `git lfs ls-files | head` (shows tracked files). CI needs `LFS_DEPLOY_KEY` in stellcoilbench repo secrets for runs that add submissions.

## Adding Submissions to Your Fork

By default, your fork's `.lfsconfig` points to the upstream LFS repo. **You cannot push to it** (no write access). To add your own submissions from your fork:

### Option A: Use your own GitHub LFS repo (simplest)

1. Create a new repo under your account (e.g. `yourname/stellcoilbench-lfs-storage`).
2. Override `.lfsconfig` in your fork:

```ini
[lfs]
    url = https://github.com/yourname/stellcoilbench-lfs-storage.git/info/lfs
```

3. Commit and push `.lfsconfig` to your fork.
4. Push as normal. LFS objects go to your repo; your quota is used, not upstream's.

### Option B: S3 or other LFS server

1. Set up an LFS backend (S3 + LFS server, GCS, R2, etc.).
2. Add `.lfsconfig` to your fork:

```ini
[lfs]
    url = https://your-lfs-server.example.com
```

3. Commit and push. Ensure your server accepts authenticated pushes from your CI/local Git.

## Fork CI

- **Without submissions:** Use `GIT_LFS_SKIP_SMUDGE=1` or sparse checkout excluding `submissions/`.
- **With submissions:** Pull LFS (`git lfs pull`) and use your own LFS backend (Option A or B above) so you do not hit upstream's quota.

## Contributing to Upstream

PRs that add or change submissions are typically not merged from forks; upstream CI is the main source. For code-only changes, standard PR workflow applies — no LFS setup needed if you use a code-only clone.
