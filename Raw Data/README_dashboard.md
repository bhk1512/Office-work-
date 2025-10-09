# 📘 Dashboard Overview & Data Rules

## Executive Summary
This dashboard unifies **tower erection progress** and **microplan execution** across projects to enable  
**early risk detection, proactive recovery, and accountable governance**.

**At a glance (Last update: {{ last_update_ist }})**
- Achievement (Revenue): **{{ kpi_achievement_rev }}**
- Achievement (MT): **{{ kpi_achievement_mt }}**
- Data Health: **{{ data_health_clean_pct }}% clean**, **{{ data_health_fallback_pct }}% fallback**
- Off-track Projects: **{{ projects_offtrack_count }}**
- Data Exceptions This Week: **{{ exceptions_count }}**

---

## What Leaders Can Do Here
- **Identify at-risk projects** by achievement gap and potential loss.
- **Prioritize interventions** based on revenue or tower-weight shortfalls.
- **Validate execution narratives** by comparing realised revenue vs. physical progress.
- **Hold ownership reviews** with transparent data lineage and logged exceptions.

---

## KPI Definitions
| Metric | Definition | Notes |
|--------|-------------|-------|
| **Achievement (Revenue)** | Delivered revenue ÷ Planned revenue | ₹ format with thousands separators |
| **Achievement (MT)** | Delivered tower weight ÷ Planned tower weight | Values in Metric Tonnes |
| **Estimated Loss** | Gap vs baseline productivity (capped at 15 days) | Uses 5 MT/day minimum baseline |

---

## Data Refresh & Coverage
- **Refresh Cadence:** Daily at *{{ refresh_time }} IST* (previous day cutoff: 23:59 IST)
- **Data Source:** Site-level Excel files emailed from field teams
- **Coverage:** Projects active since **Jan 1 2021**
- **Scope:** All erection and microplan files across pan-India operations
- **Out-of-scope:** Missing or malformed files appear in `Data Issues`

---

## Guardrails (By Design)
- **Loss Cap:** Maximum of 15 days per project to avoid inflated losses.  
- **Baseline:** 5 MT/day minimum when history is insufficient.  
- **Future Completions:** Rows with completion ≥ today are **excluded** from expansion.  
- **Single-Occurrence Gangs:** Ignored from expansion to remove one-off noise.  
- **Header Requirements:** `Status`, `Start Date`, `Completion Date`, `Gang`, `Tower Weight`, `Location No.`  
- **Project Naming:** Inferred from file name (e.g., `TA123_March.xlsx` → TA123).  
- **Normalization:** Gang names cleaned (Title Case, digits split), projects case-insensitive.  
- **Location Matching:** Normalized to lowercase, trimmed, removes trailing `.0`; `12/0` ≠ `12/0A`.

---

## Limitations & Bias Notes
- **Fallback Mode:** When `DailyExpanded` is missing, delivered revenue = realised revenue only (may bias results).  
- **Tower Weight “Optional” Field:** Required in practice; missing values block ingestion.  
- **Contiguous Data Rule:** Rows after the first blank line are ignored (notes beyond that are dropped).  
- **File Dependency:** Project code relies on consistent filename patterns.  
- **Bias Awareness:** Metrics may under- or over-represent progress when fallback exceeds 10% of rows.

---

## Data Quality Rules (Summary)
- **Date Cutoff:** Start < 2021-01-01 → excluded to `Data Issues`.  
- **Tower Weight Range:** 10–200 MT valid; out-of-range → logged.  
- **Duration Validity:** Missing dates or Start > End → logged.  
- **Gang Filter:** Gangs appearing once → not expanded.  
- **Future Completions:** Completion ≥ today → logged.  
- **Outputs:**  
  - *ErectionCompiled_Output.xlsx* → `DailyExpanded`, `RawData`, `Data Issues`, `Issues`, `Diagnostics`, `ProjectDetails`, `README_Assumptions`  
  - *MicroPlanCompiled_Output.xlsx* → `MicroPlanResponsibilities`, `MicroPlanIndex`

---

## Ownership & Escalation
| Item | Owner | SLA |
|------|--------|-----|
| Data ingestion | {{ data_owner_name }} | Within {{ ingestion_sla_hours }} hours of file receipt |
| Quality issue resolution | {{ data_owner_name }} | Within {{ quality_fix_hours }} hours |
| Governance flag | Auto-flag if > {{ quality_flag_pct }}% rows hit fallback/`Data Issues` |

---

## Usage Cadence
- **Weekly:** Review site-level performance & loss summary  
- **Monthly:** Governance meeting → validate top loss drivers & recovery actions  
- **Ad-hoc:** Exception alerts for delayed or missing data feeds  

---

## Change Log
| Date | Update |
|------|---------|
| 2025-10-09 | Added leadership-friendly summary and KPI injection support |
| 2025-10-09 | Documented future-completion and fallback rules |
| 2025-10-09 | Aligned sheet naming with actual outputs (`MicroPlanResponsibilities`) |

---

*Version: v1.0 | Generated on {{ last_update_ist }} | Maintained by {{ maintainer_name }}*
