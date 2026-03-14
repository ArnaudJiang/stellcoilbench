# Pre-Migration Checklist: Submissions to Git LFS

Complete these steps **before** running `tools/ci_lfs_migrate.sh`. The migration rewrites history and requires force-push.

---

## 1. Create the dedicated LFS repository

On GitHub, create a new repository:

- **Name:** `stellcoilbench-lfs` (or match the URL in `.lfsconfig`)
- **Owner:** `akaptano` (or your org)
- **Visibility:** Public (so forks can pull; only you can push)
- **Contents:** Empty – no need for a README or other files

The repository is used only for LFS object storage.

---

## 2. Verify `.lfsconfig` URL

Ensure `.lfsconfig` points to your LFS repo via the SSH host alias:

```ini
[lfs]
    url = git@github-lfs:akaptano/stellcoilbench-lfs.git
```

The host `github-lfs` is configured by CI to use the LFS deploy key. If your org or repo name differs, update accordingly.

---

## 3. CI authentication: second deploy key for LFS

**GitHub does not allow the same deploy key on multiple repositories.** The main repo (`stellcoilbench`) keeps its existing deploy key; LFS uses a separate deploy key for `stellcoilbench-lfs`.

### Generate the LFS deploy key

```bash
ssh-keygen -t ed25519 -C "stellcoilbench-lfs-deploy" -f ~/.ssh/stellcoilbench_lfs_deploy -N ""
```

### Add to stellcoilbench-lfs

1. Open **stellcoilbench-lfs** → Settings → Deploy keys → Add deploy key.
2. Paste the **public** key (`~/.ssh/stellcoilbench_lfs_deploy.pub`).
3. Enable **Allow write access**.
4. Title: `CI LFS deploy`.

### Add to stellcoilbench

1. Open **stellcoilbench** → Settings → Secrets and variables → Actions.
2. New repository secret: name `LFS_DEPLOY_KEY`.
3. Value: entire contents of the **private** key (`~/.ssh/stellcoilbench_lfs_deploy`).

### How it works

CI runs `tools/ci_setup_lfs_ssh.sh` before each push. It writes the key to a temp file and adds an SSH config entry for host `github-lfs` that uses that key. The main repo push still uses the existing `DEPLOY_KEY`; LFS pushes use `LFS_DEPLOY_KEY` via the host alias.

---

## 4. Back up the repository

Before rewriting history:

```bash
# Option A: Create a backup branch
git branch backup-pre-lfs-migration

# Option B: Clone to a backup directory
cp -a /path/to/stellcoilbench_clone /path/to/stellcoilbench_backup
```

---

## 5. Coordinate with collaborators

- Migration changes all commit SHAs.
- Collaborators must re-clone or run:  
  `git fetch origin && git reset --hard origin/main`
- Open branches and PRs will diverge and likely need to be recreated.

---

## 6. Install Git LFS locally

```bash
brew install git-lfs   # macOS
git lfs install
```

---

## 7. Run the migration

```bash
bash tools/ci_lfs_migrate.sh
```

Then:

```bash
git push --force-with-lease origin main
```

---

## Quick checklist

- [ ] `stellcoilbench-lfs` repo exists on GitHub
- [ ] `.lfsconfig` URL matches your LFS repo
- [ ] CI deploy key/PAT has write access to `stellcoilbench-lfs`
- [ ] Backup branch or directory created
- [ ] Collaborators notified
- [ ] `git lfs` installed
- [ ] Migration run and force-push completed
