# Project Handover

Last reviewed: 2026-07-22

## Executive Summary

This repository runs a Python-based DPR productivity system for transmission-line work. It has three main responsibilities:

1. Pull latest DPR Excel attachments from Outlook.
2. Compile raw DPR and micro-plan workbooks into normalized Excel/parquet datasets.
3. Serve a Dash dashboard and generate daily DPR summary mail artifacts.

The project is operational but needs a structured handover because the business logic is encoded across large Python modules and project-specific Excel schema assumptions. The next maintainer does not need to be an expert coder on day one, but must be disciplined about running known commands, checking generated diagnostics, and using AI against small, well-scoped code areas.

## Current State

- Branch: `main`
- Main remotes:
  - `origin`: GitHub repository
  - `vm`: production VM repository at `ercprdadmin@10.10.120.94`
- Python runtime in this working copy: `venv\Scripts\python.exe`
- System `python` is not currently available on PATH in this machine; use the repo venv or `run_app.bat`/`run_pipeline_commit.ps1`.
- Latest local validation:
  - `venv\Scripts\python.exe -m pytest`
  - Result: 137 passed, 1 failed, 7 warnings
  - Failure: `tests\test_executive_project_overview.py::ExecutiveProjectOverviewTests::test_overview_and_modal_dpr_strip_share_union_fields`
  - Nature: UI strip label expectation mismatch. Actual labels are `Project`, `Latest DPR Date`, `Scope`, `Number of TSE`, `RAG`; expected labels include `Voltage`, `Scope (km)`, and `DPR Staleness`.
- Current working tree contains live data changes from the latest DPR refresh. Before handover, decide whether these generated data updates should be committed or reverted.

## Mental Model

Think of this as a data product, not a pure web app.

```text
Outlook email attachments
        |
        v
Raw Data/DPRs/*.xlsx
        |
        v
pipeline_runner.py
        |
        +--> Parquets/Erection/*
        +--> Parquets/Stringing/*
        +--> Parquets/Foundation/*
        +--> Parquets/ProgressStatus/*
        +--> Parquets/StretchReadiness/*
        +--> Parquets/StringingSummary/*
        |
        v
app.py + dashboard/*
        |
        +--> Dash dashboard on port 8050
        +--> Daily DPR mail HTML/draft
```

The hardest part is not Dash. The hard part is normalizing inconsistent Excel formats from different projects while preserving business meaning.

## Main Daily Commands

Run these from the repository root.

### Start Dashboard Locally

```powershell
.\run_app.bat
```

What it does:

- Creates/uses `venv`.
- Installs `requirements.txt`.
- Runs `outlook_dpr_watcher.py` unless `SKIP_DPR_WATCHER=1`.
- Runs `pipeline_runner.py --config pipeline_config.json --no-serve`.
- Serves `app:server` using Waitress on `0.0.0.0:8050`.

To skip Outlook pull:

```powershell
$env:SKIP_DPR_WATCHER="1"
.\run_app.bat
```

### Refresh Pipeline Without Serving

```powershell
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --no-serve
```

Useful scoped runs:

```powershell
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --scope erection --no-serve
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --scope stringing --no-serve
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --scope foundation --no-serve
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --scope both --no-serve
```

Force stringing rebuild when cached stringing outputs look stale:

```powershell
.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --force-stringing-rebuild --no-serve
```

### Pull DPR Attachments From Outlook

```powershell
.\venv\Scripts\python.exe outlook_dpr_watcher.py
```

Continuous watch mode:

```powershell
.\venv\Scripts\python.exe outlook_dpr_watcher.py --watch
```

This requires Windows Outlook and `pywin32`.

### Prepare Daily DPR Mail

```powershell
.\venv\Scripts\python.exe prepare_daily_dpr_mail.py
```

Generate HTML without creating an Outlook draft:

```powershell
.\venv\Scripts\python.exe prepare_daily_dpr_mail.py --no-draft
```

Use existing parquet/workbook outputs:

```powershell
.\venv\Scripts\python.exe prepare_daily_dpr_mail.py --skip-refresh --no-draft
```

### Scheduled Commit/Deploy Job

```powershell
.\run_pipeline_commit.ps1
```

What it does:

- Ensures `venv`.
- Installs dependencies.
- Runs Outlook watcher.
- Runs the pipeline.
- Runs `git add -A`.
- Commits if there are staged changes.
- Pushes to `vm main`.
- Pushes to `origin main`.

Do not run this casually. It commits generated artifacts and pushes to production-facing remotes.

Scheduler wrapper:

```powershell
.\dashboard_scheduler.bat
```

Configured trigger times in the batch file:

- `10:30`
- `11:30`
- `12:02`
- `16:00`

## Repository Map

### Entry Points

- `app.py`: Dash app factory, global data store, health endpoint, production security middleware.
- `pipeline_runner.py`: Orchestrates all compile stages and optionally loads/serves dashboard.
- `outlook_dpr_watcher.py`: Pulls DPR attachments from Outlook into `Raw Data/DPRs`.
- `prepare_daily_dpr_mail.py`: Refreshes outputs and builds Outlook draft/HTML body.
- `run_app.bat`: Windows local/prod-like launcher using Waitress.
- `run_pipeline_commit.ps1`: Automated refresh, commit, and push workflow.
- `dashboard_scheduler.bat`: Simple long-running Windows scheduler wrapper.

### Dashboard Package

- `dashboard/layout.py`: Dash layout/navigation.
- `dashboard/callbacks.py`: Main callback registration and UI behavior. This is the largest risk area because it is very large.
- `dashboard/state.py`: Data store/cache layer for dashboard datasets.
- `dashboard/data_loader.py`: Reads normalized workbook/parquet artifacts for the app.
- `dashboard/metrics.py`, `dashboard/charts.py`: KPI and chart helpers.
- `dashboard/analytics.py`, `dashboard/analytics_layout.py`: Erection analytics.
- `dashboard/stringing_analytics.py`, `dashboard/stringing_analytics_layout.py`: Stringing analytics.
- `dashboard/workbook.py`: Excel export/report generation logic. This is another large risk area.
- `dashboard/*_ingest.py`: Domain-specific ingestion for foundation, progress status, stretch readiness, stringing, and stringing summary.
- `dashboard/services/responsibilities.py`: Micro-plan responsibilities loading/fallbacks.

### Data and Config

- `Raw Data/DPRs`: Latest source DPR workbooks.
- `Raw Data/DPR_Config.xlsx`: Project-specific sheet/template mapping. This is a critical business configuration file.
- `Raw Data/Email_config.xlsx`: Outlook sender/attachment matching configuration.
- `Raw Data/DPR_Mail_Overrides.csv`: Manual mail output overrides.
- `Raw Data/Completed Projects.xlsx`: Project exclusion source.
- `Raw Data/Projects and PCH.xlsx`: PCH/region mapping.
- `Raw Data/Projects_KV.xlsx`: Voltage mapping.
- `Parquets/*`: Compiled datasets consumed by dashboard and mail/export scripts.
- `Productivity Summaries`: Generated reports and historical analysis outputs.
- `DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md`: Existing AI-friendly data/schema reference. This should be treated as a first-read document for any AI session.

### Tests

- `tests`: Unit tests for ingestion, exports, mail generation, dashboard layout/callback behavior, and pipeline scope selection.
- Use `venv\Scripts\python.exe -m pytest` before pushing logic changes.
- If a change only touches one domain, run a targeted file first, then the full suite.

## Data Outputs Snapshot

Current compiled parquet row counts:

- Erection:
  - `ProdDailyExpandedSingles.parquet`: 46,026 rows
  - `RawData.parquet`: 5,580 rows
  - `MicroPlanResponsibilities.parquet`: 1,950 rows
- Foundation:
  - `FoundationRaw.parquet`: 6,353 rows
  - `FoundationCompletions.parquet`: 5,301 rows
- Progress status:
  - `RawData.parquet`: 4,871 rows
- Stretch readiness:
  - `RawData.parquet`: 1,604 rows
  - `Summary.parquet`: 10 rows
- Stringing:
  - `StringingCompiled.parquet`: 1,272 rows
  - `StringingDaily.parquet`: 8,636 rows
  - `Data Issues.parquet`: 990 rows
- Stringing summary:
  - `StatusActivityFact.parquet`: 4,623 rows
  - `ManpowerProductivityFact.parquet`: 8,636 rows
  - `StretchSectionFact.parquet`: 1,604 rows

These counts are useful as a quick sanity baseline after a refresh. Exact values will change as new DPRs arrive.

## Critical Business Rules

- Completed projects are excluded using `Raw Data/Completed Projects.xlsx`.
- Project names and line names are normalized heavily; do not change project identity logic casually.
- `Raw Data/DPR_Config.xlsx` drives many project-specific sheet and template mappings.
- The system must handle malformed/manual Excel files. Several loaders intentionally use fallback strategies.
- Erection, stringing, foundation, progress status, stretch readiness, and summary outputs are separate but connected.
- Stringing summary depends on previously compiled Stringing, ProgressStatus, and StretchReadiness outputs.
- Future completion rows are intentionally excluded or flagged depending on context.
- Dashboard reads mostly from `Parquets/*`, not directly from raw Excel files.

## Current Risks

### High

- `dashboard/callbacks.py` is over 16k lines. Changes here are hard to review and easy to regress.
- `dashboard/workbook.py` is over 8k lines. Export behavior is dense and business-rule-heavy.
- Generated artifacts, logs, pycache, and local cache files are tracked despite `.gitignore` rules. This makes handover and code review harder.
- `run_pipeline_commit.ps1` runs `git add -A` and pushes to both remotes. A bad local state can be committed automatically.
- Windows/Outlook dependency means some flows cannot be reproduced on a non-Windows server without alternatives.

### Medium

- `README.md` is too short for a new maintainer.
- Config is split across JSON, Excel, environment variables, and batch/PowerShell scripts.
- `.env.example` does not mention all environment variables used by `dashboard/config.py`.
- Several generated investigation scripts and one-time exports remain at repository root.
- The latest test run has one failing UI test.

### Low

- `.claude/settings.local.json` exists and appears user/machine-specific.
- `DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md` is useful but generated on 2026-04-29 and may be stale against current data.

## Recommended Handover Plan

### Day 0: Freeze and Package

1. Decide whether the current uncommitted DPR/data updates are correct.
2. Commit only the intended final state.
3. Tag or note a known-good handover commit.
4. Export/share non-Git operational context:
   - VM access method
   - Windows scheduler/task setup
   - Outlook account/folder expectations
   - Email recipients for DPR draft
   - Who owns each raw workbook source
5. Share this file, `README.md`, `DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md`, and the latest passing/failing test status.

### Week 1: Operator Onboarding

The new maintainer should learn operations before editing code.

1. Clone repo and confirm `venv`.
2. Run `.\venv\Scripts\python.exe -m pytest`.
3. Run `.\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --no-serve`.
4. Run `.\venv\Scripts\python.exe prepare_daily_dpr_mail.py --skip-refresh --no-draft`.
5. Start app with `.\run_app.bat`.
6. Open dashboard and compare key KPIs with latest DPR/mail.
7. Read diagnostics parquet/workbook sheets before debugging raw code.

### Week 2: Safe Maintenance

1. Fix the current failing UI test or intentionally update its expectation.
2. Add/update documentation for common issues:
   - Missing DPR attachment
   - New project onboarding
   - New DPR sheet format
   - Dashboard not starting
   - Outlook watcher not saving files
3. Create a simple release checklist before running `run_pipeline_commit.ps1`.
4. Split generated outputs from source code, or at least document which generated files are expected to change daily.

### Month 1: Technical Debt Reduction

1. Split `dashboard/callbacks.py` by dashboard page/domain.
2. Split `dashboard/workbook.py` by export type.
3. Replace machine-specific scheduler paths with configurable paths.
4. Refresh `DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md`.
5. Add a compact `docs/operations.md` and `docs/new-project.md`.
6. Consider moving large generated reports and daily raw files to a storage location instead of Git.

## AI Usage Protocol For The New Maintainer

Use AI as a pair-programmer, not as an unchecked editor.

### Recommended Model Setup

- Claude/Gemini Pro are sufficient for day-to-day explanation and code navigation.
- A ChatGPT subscription is useful if the maintainer wants another strong coding assistant, but not required if Claude/Gemini are already available.
- For coding work, use an agent that can read the local repo and run tests. Browser-only chat is much less effective.

### First Prompt For Any AI Session

```text
You are helping maintain this Python Dash DPR productivity repository.
First read PROJECT_HANDOVER.md, README.md, DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md, pipeline_runner.py, app.py, and the specific test/file related to my task.
Do not edit files until you explain the likely impact and the targeted validation command.
Keep changes minimal.
```

### Debugging Prompt Template

```text
Problem: <specific symptom>
Command run: <exact command>
Output/error: <paste relevant output>
Expected behavior: <what should have happened>
Files likely involved: <if known>

Please trace the data flow, identify the smallest likely root cause, propose a minimal patch, and tell me the exact pytest command to validate it.
```

### Safe Edit Rules

- Never ask AI to “clean up the project” broadly.
- Never accept changes across `dashboard/callbacks.py` and `dashboard/workbook.py` without a targeted reason.
- Always run a targeted test before a full test.
- Before `run_pipeline_commit.ps1`, inspect `git status --short`.
- If raw DPR or parquet files changed unexpectedly, understand why before committing.

## Common Troubleshooting

### Dashboard Does Not Start

1. Run:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Run:

   ```powershell
   .\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --no-serve
   ```

3. Then run:

   ```powershell
   .\run_app.bat
   ```

### Outlook Pull Fails

- Confirm Outlook desktop is installed and configured.
- Confirm `Raw Data/Email_config.xlsx` has sender/attachment rules.
- Run:

  ```powershell
  .\venv\Scripts\python.exe outlook_dpr_watcher.py
  ```

### New Project Or Sheet Format Is Missing

1. Update `Raw Data/DPR_Config.xlsx`.
2. Re-run scoped pipeline:

   ```powershell
   .\venv\Scripts\python.exe pipeline_runner.py --config pipeline_config.json --scope all --no-serve
   ```

3. Inspect the relevant `Diagnostics`, `Issues`, and `Coverage` outputs under `Parquets`.
4. Add or update tests if code changes were required.

### Tests Fail After UI Change

Start with the exact failing test:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_executive_project_overview.py
```

If it is an intentional UI change, update expectations in the relevant test. If not, fix the callback/layout code.

## What To Explain Live During Handover

Spend 60-90 minutes walking through:

1. Where DPR emails come from and how attachments are named.
2. How `Raw Data/DPR_Config.xlsx` maps messy workbooks into normalized tables.
3. How to run the pipeline manually.
4. How to inspect `Diagnostics`, `Issues`, and `Coverage`.
5. How to start the dashboard.
6. How daily mail generation works.
7. How automated commit/push works and when not to use it.
8. How to ask AI for small, testable changes.

## Immediate Action Items Before Leaving

1. Resolve or document the failing `test_overview_and_modal_dpr_strip_share_union_fields` test.
2. Clean the final Git state: commit intended DPR/output changes or revert them.
3. Document VM credentials/access handoff outside Git.
4. Document Outlook folder/account assumptions outside Git.
5. Confirm whether generated Excel/parquet/log/cache files should stay tracked.
6. Refresh `DATA_SCHEMA_REFERENCE_FOR_CLAUDE.md` if it is still used as the AI schema source.
7. Give the new maintainer one dry-run task: regenerate daily mail with `--skip-refresh --no-draft`.

