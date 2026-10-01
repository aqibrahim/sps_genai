#!/usr/bin/env bash
# Creates the local git repo for sps_genai and pushes it to GitHub.
# Run from inside the sps_genai folder:   bash setup_github.sh
set -e

REPO_NAME="sps_genai"

command -v git >/dev/null || { echo "Git is not installed. Install it from https://git-scm.com/downloads"; exit 1; }

# Make sure git knows who you are (needed for commits)
if [ -z "$(git config --global user.name)" ]; then
  read -rp "Your name (for git commits): " NAME;  git config --global user.name "$NAME"
fi
if [ -z "$(git config --global user.email)" ]; then
  read -rp "Your GitHub email: " EMAIL;            git config --global user.email "$EMAIL"
fi

# 1) Local repository with two clear commits
if [ ! -d .git ]; then
  git init -b main
  git add .gitignore .python-version README.md
  git commit -m "Initial project setup (Module 1)"
  git add .
  git commit -m "Assignment 1: add spaCy word embedding endpoints, bigram API and Docker deployment"
else
  echo "Git repo already exists, committing any changes..."
  git add . && git commit -m "Assignment 1 updates" || true
fi

# 2) Create the GitHub repo and push
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  echo; echo "Done! Your repo: $(gh repo view --json url -q .url)"
else
  echo
  echo "Local repo is ready. GitHub CLI not found/not logged in, so finish manually:"
  echo "  1. Go to https://github.com/new, name it '$REPO_NAME', choose Public,"
  echo "     and do NOT add a README/.gitignore/license. Click 'Create repository'."
  echo "  2. Then run (replace YOUR-USERNAME):"
  echo "       git remote add origin https://github.com/YOUR-USERNAME/$REPO_NAME.git"
  echo "       git push -u origin main"
fi
