# Team Workflow Guide (Clear + Fixed)

Repository: https://github.com/PixelA42/project_s.a.f.e

This is the exact workflow for all 3 members.

## Member Roles

- Member 1: Core ML (only person allowed to work directly on `main`)
- Member 2: Data Pipeline + Backend member (branch-only)
- Member 3: UI + Testing member (branch-only)

## One Rule For Everyone

- Member 2 and Member 3 must never push directly to `main`.
- They must create branch, push branch, open PR, then merge only after review.

## Fixed Branches and Tasks

Use these exact branch names so everything stays clean.

### Member 2 (Data Pipeline + Backend)

Branch 1: `member2/data-split-manifest`

Tasks:
- Dataset split logic (as per research)
- Distress keyword CSV setup
- Dataset manifest JSON generation

Branch 2: `member2/backend-api-validation`

Tasks:
- API request validation
- Audio file checks (format/size)
- Error response structure

Branch 3: `member2/backend-calllog-persistence`

Tasks:
- DB model for call logs
- Persist risk evaluation output
- DB constraints for score ranges

### Member 3 (UI + Testing)

Branch 1: `member3/ui-risk-states`

Tasks:
- Loading state
- Safe state
- Prank state
- Scam likely state

Branch 2: `member3/ui-api-integration`

Tasks:
- Connect UI with backend endpoints
- Loading/error handling in UI
- Show final/spectral/intent scores

Branch 3: `member3/tests-property-smoke`

Tasks:
- Add/maintain pytest test coverage
- Property-style checks for score ranges
- Endpoint smoke tests

## Setup Once (for Member 2 and Member 3)

```bash
git clone https://github.com/PixelA42/project_s.a.f.e.git
cd project_s.a.f.e
git checkout main
git pull origin main
```

## Exact Work Steps (for Member 2 and Member 3)

### Step 1: Create your task branch

Example for Member 2:

```bash
git checkout main
git pull origin main
git checkout -b member2/data-split-manifest
```

Example for Member 3:

```bash
git checkout main
git pull origin main
git checkout -b member3/ui-risk-states
```

### Step 2: Commit your work

```bash
git add .
git commit -m "feat: implement assigned task"
```

### Step 3: Push your branch

```bash
git push -u origin member2/data-split-manifest
```

or

```bash
git push -u origin member3/ui-risk-states
```

### Step 4: Open PR to main

- PR base: `main`
- PR compare: your branch
- Add short PR note: what changed, why, and test result

### Step 5: Merge only after review and checks

- Do not self-merge if checks are failing
- Do not merge mixed/unrelated work

## Core ML Member (Member 1) Process

Core ML member can work on `main`, but should still follow safe practice:

```bash
git checkout main
git pull origin main
# do work
git add .
git commit -m "feat: core ml update"
git push origin main
```

## Minimum Check Before Any Merge

- Tests pass: `pytest tests/ -v --tb=short`
- No broken lint/format in changed files
- PR scope is only one task

## Simple Do / Do Not

Do:
- Member 2 and Member 3: always branch first
- Keep one branch = one task
- Rebase/merge latest main before final PR merge

Do Not:
- Member 2 or Member 3 pushing to `main`
- One PR containing many unrelated tasks
- Merge when tests are red
