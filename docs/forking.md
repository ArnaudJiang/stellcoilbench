# Forking StellCoilBench

This guide explains how to fork StellCoilBench and work with Git LFS, which is used for large files in `submissions/` and `cases/done/` (VTK meshes, plots, JSON).

## LFS on Main

Submissions and autopilot results use **Git LFS** with storage in the main repository. When you push to your fork, LFS objects go to your fork's LFS storage by default; upstream's quota is not used.

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

### Verifying LFS

After cloning, confirm LFS objects are present: `git lfs fsck` (no missing/deleted) and `git lfs ls-files | head` (shows tracked files).

### Upstream CI and unrecoverable LFS objects

Upstream CI sets `GIT_LFS_SKIP_SMUDGE=1` so that LFS objects are never fetched during checkout. This workaround exists because some historic LFS OIDs may be unrecoverable (404) and would cause `git lfs pull` to fail. CI jobs do not need the blob content: leaderboard generation reads `results.json` and `.zip` files (not LFS), and benchmark/autopilot jobs only create new output. LFS push still works when committing new submissions. `.lfsconfig` sets `skipdownloaderrors = true` so fetches continue on 404.

## Adding Submissions to Your Fork

When you push to your fork, LFS objects are stored in your fork's LFS storage by default. No `.lfsconfig` override is needed.

If you want to use a separate LFS backend (e.g. to isolate quota or use S3):

1. Create a new repo under your account (e.g. `yourname/stellcoilbench-lfs-storage`), or set up S3/GCS/R2 with an LFS server.
2. Add `.lfsconfig` to your fork:

```ini
[lfs]
    url = https://github.com/yourname/stellcoilbench-lfs-storage.git/info/lfs
```

3. Commit and push `.lfsconfig` to your fork.
4. Push as normal. LFS objects go to your configured backend.

## Fork CI

- **Without submissions:** Use `GIT_LFS_SKIP_SMUDGE=1` or sparse checkout excluding `submissions/`.
- **Upstream CI:** Uses `GIT_LFS_SKIP_SMUDGE=1` because some historic LFS objects may be unrecoverable (404). CI-critical paths (`results.json`, `.zip`) are not LFS, so jobs run correctly with pointer files. LFS push for new submissions uses `DEPLOY_KEY` (main repo).
- **With submissions:** Push to your fork; LFS goes to your fork's storage. Or use your own LFS backend via `.lfsconfig`.

## Contributing to Upstream

PRs that add or change submissions are typically not merged from forks; upstream CI is the main source. For code-only changes, standard PR workflow applies — no LFS setup needed if you use a code-only clone.
