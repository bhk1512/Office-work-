# Data Structure Reference for Claude

Generated on: 2026-04-29 11:27:18
Workspace: `Office-work-`

## Scope
This file documents the Excel data structure needed for downstream analysis without sharing full source files.
It includes:
- Sheet names for each DPR workbook under `Raw Data/DPRs`.
- Sheet names + detected column headers for core input/output workbooks.
- DPR_Config mappings for stringing/status/stretch pipelines.
- 10-row samples for new datasets: stringing manpower, stretch readiness, and progress status.
- CSV inputs in repo: none (excluding `venv`).

## Core Workbook Schemas (Sheet Names + Column Headers)

### Raw Data/DPR_Config.xlsx
- Sheets (33): Sheet Names Check, TA 413 Erection, TA 413 Stringing, TA 601 Stringing, TB 501 220kV Stringing, TB 501 132kV Stringing, TB 501 220kV Erection, TB 501 132kV Erection, TB 401 Stringing, TB 408 Stringing, TA 509 Erection, TA 601 Erection, TB 408 Status, TA 413 Status, TA 414 Status, TA 416 Status, TA 418 Status, TA 421 Status, TA 419 Status, TA 504 Status, TA 505 Status, TA 506 Status, TB 507 400kV Status, TB 507 765kV Status, TB 501 Status, TA 413 Stretch, TA 418 Stretch, TA 419 Stretch, TA 421 Stretch, TA 504 Stretch, TA 505 Stretch, TA 506 Stretch, TB 507 Stretch
- `Sheet Names Check` headers (15, header row 1): Project Code, Erection Sheet Names, Stringing Sheet Names, Erection Template Check, Stringing Template Check, Erection Line Names, Stringing Line Names, Status Sheet Names, Status Template Check, Status Line Names, Stretch Readiness Sheet Names, Stretch Daily Stringing Sheet Names, Stretch Template Check, Stretch Line Names, Stretch Manpower Expected
- `TA 413 Erection` headers (13, header row 2): CUMM. SR.NO, MONTH SR.NO., Loc. No., Type of Tower, Start, Complete, Gang name, Weight, Revenue, Name of Supervisor, Remarks, JMC Status, Prop. Name
- `TA 413 Stringing` headers (21, header row 3): CUMM. SR.NO, MONTH SR.NO., From, To, Length in Km's, Cum. Length in Km's, TSE/Manual, Insulator hoisting Start, Insulator Hoisting Complete, Paying Out Start, Paying Out Completed, Final sag start, Final sag complete, Earthwire Final Sag, Jumpuring, Spacering, Clipping, Revenue, JMC Status, Name of the Contractor, Remarks
- `TA 601 Stringing` headers (23, header row 3): S.no, Sec, From AP, To AP, Span (M), Method, Section Readiness, Insulator Hoisting, P/O Starting Date, P/O Completion Date, P/O, F/S Starting Date, F/S/ Completion Date, Jumpering, Spacering, Clipping, Length, Status, Gang Name, Gang Strength, Number of Fitters, Hindrance, Remarks
- `TB 501 220kV Stringing` headers (24, header row 4): unnamed_col_1, From, To, unnamed_col_4, unnamed_col_5, unnamed_col_6, Status, DOS, DOC, Status__2, DOS__2, DOC__2, Status__3, Status__4, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, Status__5, DOS__3, DOC__3, Status__6, DOS__4, DOC__4
- `TB 501 132kV Stringing` headers (25, header row 4): unnamed_col_1, From, To, unnamed_col_4, unnamed_col_5, unnamed_col_6, Status, DOS, DOC, Status__2, DOS__2, DOC__2, Status__3, Status__4, Status__5, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, Status__6, DOS__3, DOC__3, Status__7, DOS__4, DOC__4
- `TB 501 220kV Erection` headers (65, header row 3): Sl. No., LOC NO, Tower Type, unnamed_col_4, unnamed_col_5, unnamed_col_6, unnamed_col_7, A, B, C, D, HT, MS, pack Washar, BnA, Total Tower Weight, HT__2, BnA__2, Total Stub Weight, Deviation of Angle, State, Span Length(m), Classification, Excavation Volume (m3), Concreting Volume, unnamed_col_26, unnamed_col_27, Cement Required, unnamed_col_29, Total Steel (Kgs) per Tower, unnamed_col_31, unnamed_col_32, unnamed_col_33, unnamed_col_34, unnamed_col_35, unnamed_col_36, unnamed_col_37, unnamed_col_38, Fnd Status
(C/WIP), Manual Checklist, JMC, Excavation, unnamed_col_43, Concreting, unnamed_col_45, Sub-Contractor Name, unnamed_col_47, Earthing Status
C/WIP, Earthing Type, DOS, DOC, S/C, unnamed_col_53, Erection Status
C/WIP, Manual Checklist__2, JMC__2, DOS__2, DOC__2, S/C__2, unnamed_col_60, Status
C/WIP, DOS__3, DOC__3, NB qty, Sub-Contractor Name__2
- `TB 501 132kV Erection` headers (52, header row 6): Sl. No., LOC NO, Tower Type, unnamed_col_4, unnamed_col_5, unnamed_col_6, Tower Weight, unnamed_col_8, unnamed_col_9, unnamed_col_10, Stub Weight, unnamed_col_12, unnamed_col_13, unnamed_col_14, Chimney Extension, unnamed_col_16, unnamed_col_17, unnamed_col_18, Span Length(m), Classification, Excavation Volume (m3), Concreting Volume, unnamed_col_23, unnamed_col_24, 8mm Steel, 10mm Steel, 12mm Steel, 16mm Steel, 20mm Steel, 25mmSteel, Total Steel (MT), Status
(C/WIP), Excavation, unnamed_col_34, Concreting, unnamed_col_36, Sub-Contractor Name, Status
C/WIP, Type, DOS, DOC, Sub-Contractor Name__2, Status
C/WIP__2, DOS__2, DOC__2, Sub-Contractor Name__3, unnamed_col_47, Status
C/WIP__3, DOS__3, DOC__3, Painting, Sub-Contractor Name__4
- `TB 401 Stringing` headers (10, header row 4): unnamed_col_1, From, To, Length, Start Date, Finish Date, Month, Start Date__2, Finish Date__2, Month__2
- `TB 408 Stringing` headers (12, header row 7): S.no, From AP, To AP, Method, Gang Name, Span (M), P/O, P/O Starting Date, P/O Completion Date, Length, F/S Starting Date, F/S/ Completion Date
- `TA 509 Erection` headers (7, header row 2): SL NO., LOC No., STATUS, STARTING DATE, ENDING DATE, CONTRACTOR, JMC
- `TA 601 Erection` headers (10, header row 2): SL No., Location No., Tower Type, Classification, Start Date, MT, Status, End Date, Gang Name, Gang Strength
- `TB 408 Status` headers (13, header row 13): unnamed_col_1, activity_raw, quantity_estimated_or_total, cumulative_last_month, unnamed_col_5, plan_for_month, today_progress, progress_for_month, unnamed_col_9, cumulative_progress, balance_progress, gangs_working, remarks
- `TA 413 Status` headers (13, header row 13): unnamed_col_1, activity_raw, quantity_loa, quantity_estimated_or_total, cumulative_last_month, unnamed_col_6, plan_for_month, progress_for_month, today_progress, cumulative_progress, balance_progress, gangs_working, remarks
- `TA 414 Status` headers (13, header row 13): unnamed_col_1, activity_raw, quantity_estimated_or_total, cumulative_last_month, unnamed_col_5, plan_for_month, today_progress, progress_for_month, cumulative_progress, balance_progress, unnamed_col_11, gangs_working, remarks
- `TA 416 Status` headers (15, header row 13): activity_raw, unnamed_col_2, quantity_estimated_or_total, unnamed_col_4, cumulative_last_month, unnamed_col_6, plan_for_month, today_progress, progress_for_month, unnamed_col_10, cumulative_progress, balance_progress, unnamed_col_13, gangs_working, remarks
- `TA 418 Status` headers (15, header row 13): activity_raw, unnamed_col_2, quantity_estimated_or_total, unnamed_col_4, cumulative_last_month, unnamed_col_6, plan_for_month, today_progress, progress_for_month, unnamed_col_10, cumulative_progress, balance_progress, unnamed_col_13, gangs_working, remarks
- `TA 421 Status` headers (16, header row 13): unnamed_col_1, unnamed_col_2, activity_raw, quantity_estimated_or_total, unnamed_col_5, cumulative_last_month, unnamed_col_7, plan_for_month, today_progress, progress_for_month, unnamed_col_11, cumulative_progress, balance_progress, unnamed_col_14, gangs_working, remarks
- `TA 419 Status` headers (15, header row 13): unnamed_col_1, unnamed_col_2, activity_raw, quantity_loa, quantity_estimated_or_total, unnamed_col_6, cumulative_last_month, unnamed_col_8, plan_for_month, progress_for_month, today_progress, cumulative_progress, balance_progress, gangs_working, remarks
- `TA 504 Status` headers (13, header row 13): unnamed_col_1, activity_raw, unnamed_col_3, quantity_estimated_or_total, cumulative_last_month, balance_progress, plan_for_month, progress_for_month, unnamed_col_9, cumulative_progress, unnamed_col_11, unnamed_col_12, remarks
- `TA 505 Status` headers (13, header row 13): unnamed_col_1, activity_raw, unnamed_col_3, quantity_estimated_or_total, cumulative_last_month, balance_progress, plan_for_month, progress_for_month, unnamed_col_9, cumulative_progress, unnamed_col_11, unnamed_col_12, remarks
- `TA 506 Status` headers (13, header row 13): unnamed_col_1, activity_raw, unnamed_col_3, quantity_estimated_or_total, cumulative_last_month, balance_progress, plan_for_month, progress_for_month, unnamed_col_9, cumulative_progress, unnamed_col_11, unnamed_col_12, remarks
- `TB 507 400kV Status` headers (13, header row 13): unnamed_col_1, activity_raw, quantity_loa, quantity_estimated_or_total, cumulative_last_month, plan_for_month, progress_for_month, unnamed_col_8, today_progress, cumulative_progress, balance_progress, gangs_working, remarks
- `TB 507 765kV Status` headers (13, header row 13): unnamed_col_1, activity_raw, quantity_loa, quantity_estimated_or_total, cumulative_last_month, plan_for_month, progress_for_month, unnamed_col_8, today_progress, cumulative_progress, balance_progress, gangs_working, remarks
- `TB 501 Status` headers (18, header row 14): unnamed_col_1, activity_raw, unnamed_col_3, quantity_estimated_or_total, unnamed_col_5, unnamed_col_6, unnamed_col_7, cumulative_last_month, plan_for_month, today_progress, progress_for_month, unnamed_col_12, cumulative_progress, balance_progress, unnamed_col_15, gangs_working, unnamed_col_17, remarks
- `TA 413 Stretch` headers (12, header row 13): unnamed_col_1, stretch_identifier, length_m, unnamed_col_4, unnamed_col_5, unnamed_col_6, unnamed_col_7, unnamed_col_8, unnamed_col_9, balance_towers, unnamed_col_11, remarks
- `TA 418 Stretch` headers (23, header row 13): unnamed_col_1, stretch_identifier, from_ap, to_ap, length_m, unnamed_col_6, readiness_raw, unnamed_col_8, unnamed_col_9, unnamed_col_10, unnamed_col_11, unnamed_col_12, unnamed_col_13, unnamed_col_14, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, unnamed_col_20, unnamed_col_21, unnamed_col_22, remarks
- `TA 419 Stretch` headers (16, header row 13): unnamed_col_1, stretch_identifier, unnamed_col_3, length_m, unnamed_col_5, unnamed_col_6, unnamed_col_7, unnamed_col_8, readiness_raw, unnamed_col_10, section_name, unnamed_col_12, remarks, unnamed_col_14, unnamed_col_15, balance_towers
- `TA 421 Stretch` headers (22, header row 13): unnamed_col_1, stretch_identifier, from_ap, to_ap, length_m, unnamed_col_6, readiness_raw, unnamed_col_8, unnamed_col_9, unnamed_col_10, unnamed_col_11, unnamed_col_12, unnamed_col_13, unnamed_col_14, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, unnamed_col_20, unnamed_col_21, remarks
- `TA 504 Stretch` headers (23, header row 13): unnamed_col_1, stretch_identifier, from_ap, to_ap, length_m, unnamed_col_6, readiness_raw, unnamed_col_8, unnamed_col_9, unnamed_col_10, unnamed_col_11, unnamed_col_12, unnamed_col_13, unnamed_col_14, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, unnamed_col_20, unnamed_col_21, unnamed_col_22, remarks
- `TA 505 Stretch` headers (23, header row 13): unnamed_col_1, stretch_identifier, from_ap, to_ap, length_m, unnamed_col_6, readiness_raw, unnamed_col_8, unnamed_col_9, unnamed_col_10, unnamed_col_11, unnamed_col_12, unnamed_col_13, unnamed_col_14, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, unnamed_col_20, unnamed_col_21, unnamed_col_22, remarks
- `TA 506 Stretch` headers (23, header row 13): unnamed_col_1, stretch_identifier, from_ap, to_ap, length_m, unnamed_col_6, readiness_raw, unnamed_col_8, unnamed_col_9, unnamed_col_10, unnamed_col_11, unnamed_col_12, unnamed_col_13, unnamed_col_14, unnamed_col_15, unnamed_col_16, unnamed_col_17, unnamed_col_18, unnamed_col_19, unnamed_col_20, unnamed_col_21, unnamed_col_22, remarks
- `TB 507 Stretch` headers (5, header row 13): unnamed_col_1, unnamed_col_2, stretch_identifier, final_check_raw, tack_welding_raw

### Raw Data/Projects and PCH.xlsx
- Sheets (1): Sheet1
- `Sheet1` headers (4, header row 1): PCH, Project, Project Name, Region

### Raw Data/Projects_KV.xlsx
- Sheets (1): Sheet1
- `Sheet1` headers (6, header row 1): Project, Voltage, Unnamed: 2, Unnamed: 3, Unnamed: 4, Unnamed: 5

### Raw Data/Progress Summary.xlsx
- Sheets (1): Progress Summary
- `Progress Summary` headers (6, header row 1): project_code, project_name, activity, total, completed, balance

### Parquets/Stringing/StringingCompiled_Output.xlsx
- Sheets (8): Stringing Compiled, Diagnostics, Data Issues, Issues, README_Assumptions, MicroPlanResponsibilities, MicroPlanIndex, MicroPlanDataIssues
- `Stringing Compiled` headers (72, header row 1): s no, sec, from_ap, unnamed_col_4, to_ap, length_m, method, balance fdn, balance erc, section_readiness, starting date, completion date, po_start_date, po_completion_date, po, fs_starting_date, fs_complete_date, status, completion month, jmc no, gang_name, hindrance, remarks, district, project_code, line_name, project_name, project_display, project_scope_key, project, _source_file, source_sheet, method_inferred, method_inference_reason, erection_locations_for_method, S.no, Sec__2, unnamed_col_6, Insulator Hoisting, unnamed_col_9, unnamed_col_14, Jumpering, Spacering, Clipping, unnamed_col_19, Remarks__2, insulator hoisting__2, jumpering__2, spacering__2, clipping__2, gang strength, number of fitters, unnamed_col_18, revenue, tackwelding status, unnamed_col_1, unnamed_col_2, unnamed_col_3, kec international limited, insulator hosting, no of fitters, month, month__2, status__2, unnamed_col_17, status__6, dos__3, doc__3, status__7, dos__4, doc__4, status__5
- `Diagnostics` headers (21, header row 1): Workbook, Project, Sheet, ConfiguredSheet, LineName, LineNameSource, DetectedHeaderRow, ColumnsDetected, NormalizedColumnsOk, PresentColumns, MissingColumns, AppliedMap, Rows, DailyRows, Status, FallbackNote, TemplateSheet, TemplateApplied, TemplateFallbackUsed, TemplateChanges, MethodInferenceRows
- `Data Issues` headers (16, header row 1): project_name, from_ap, to_ap, gang_name, method, status, po_start_date, po_completion_date, fs_starting_date, fs_complete_date, length_m, length_km, po_km, source_file, source_sheet, Issues
- `Issues` headers (10, header row 1): Workbook, Project, Sheet, ConfiguredSheet, LineName, LineNameSource, Issue, MissingColumns, Rows, DailyRows
- `README_Assumptions` headers (2, header row 1): Note, Rules
- `MicroPlanResponsibilities` headers (17, header row 1): project_key, project_name, plan_month, entity_type, entity_name, location_no, revenue_planned, revenue_realised, tower_weight, tower_type, manpower, power_tools_issued, material_feeding, starting_date, completion_date, tack_welding, final_checking
- `MicroPlanIndex` headers (7, header row 1): file_path, project_name, project_key, rows_cleaned, status, error, plan_month
- `MicroPlanDataIssues` headers (0, header row unknown): (no columns)

### Parquets/StringingSummary/StringingSummary_Output.xlsx
- Sheets (8): StatusActivityFact, StatusSnapshotProject, StatusSnapshotOverall, StretchSectionFact, ManpowerProductivityFact, Coverage, Diagnostics, Issues
- `StatusActivityFact` headers (24, header row 1): project_code, project_display, project_scope_key, line_name, report_date, month, section_label, activity_raw, activity_norm, activity_group, core_activity, quantity_primary, cumulative_last_month, plan_for_month, progress_for_month, today_progress, cumulative_progress, balance_progress, gangs_working, remarks, source_file, source_sheet, configured_sheet, template_sheet
- `StatusSnapshotProject` headers (19, header row 1): project_code, project_display, project_scope_key, line_name, month, report_date_max, activities_total, quantity_primary_sum, cumulative_last_month_sum, plan_for_month_sum, progress_for_month_sum, today_progress_sum, cumulative_progress_sum, balance_progress_sum, completion_pct, foundation_cumulative_progress, tower_erection_cumulative_progress, stringing_cumulative_progress, opgw_stringing_cumulative_progress
- `StatusSnapshotOverall` headers (15, header row 1): month, projects_total, activities_total, quantity_primary_sum, cumulative_last_month_sum, plan_for_month_sum, progress_for_month_sum, today_progress_sum, cumulative_progress_sum, balance_progress_sum, foundation_cumulative_progress, tower_erection_cumulative_progress, stringing_cumulative_progress, opgw_stringing_cumulative_progress, completion_pct
- `StretchSectionFact` headers (22, header row 1): project_code, project_display, project_scope_key, line_name, report_date, month, section_label, section_id, from_ap, to_ap, length_km, readiness_state, is_ready, is_partial, is_not_ready, is_unknown, balance_towers, remarks, source_file, source_sheet, configured_sheet, template_sheet
- `ManpowerProductivityFact` headers (22, header row 1): project_code, project_display, project_scope_key, line_name, date, month, gang_name, from_ap, to_ap, span_key, method, section_readiness, daily_km, po_km, manpower_gang_strength, manpower_fitters, manpower_signal_type, manpower_status, expected_manpower, expected_match, availability, availability_reason
- `Coverage` headers (11, header row 1): project_code, project_display, project_scope_key, category, status, reason_code, reason, workbook, configured_sheet, resolved_sheet, rows
- `Diagnostics` headers (3, header row 1): component, status, rows
- `Issues` headers (4, header row 1): severity, component, code, message

### Parquets/StretchReadiness/StretchReadiness_Output.xlsx
- Sheets (6): RawData, Summary, ManpowerAudit, Diagnostics, Issues, Coverage
- `RawData` headers (24, header row 1): project_code, project_display, project_scope_key, line_name, line_name_source, section_label, source_file, source_sheet, configured_sheet, template_sheet, report_date, header_row_number, source_row_number, stretch_identifier, from_ap, to_ap, length_m_raw, length_km, readiness_raw, final_check_raw, tack_welding_raw, balance_towers, readiness_state, remarks
- `Summary` headers (19, header row 1): project_code, project_display, project_scope_key, line_name, line_name_source, report_date, source_files, source_sheets, total_count, ready_count, partial_count, not_ready_count, unknown_count, balance_count, total_km, ready_km, balance_km, readiness_pct, basis
- `ManpowerAudit` headers (19, header row 1): project_code, project_display, project_scope_key, line_name, line_name_source, source_file, source_sheet, configured_sheet, header_row_number, manpower_fields, readiness_fields, readiness_column_present, signal_type, non_empty_count, sample_values, expected_manpower, expected_match, status, reason
- `Diagnostics` headers (16, header row 1): Workbook, Project, Category, Sheet, ConfiguredSheet, LineName, LineNameSource, TemplateSheet, TemplateApplied, TemplateChanges, FallbackNote, SectionsDetected, HeadersDetected, Rows, Status, Reason
- `Issues` headers (9, header row 1): Workbook, Project, Category, Sheet, ConfiguredSheet, LineName, LineNameSource, Issue, Reason
- `Coverage` headers (11, header row 1): project_code, project_display, category, status, reason_code, reason, workbook, configured_sheet, resolved_sheet, rows, available_sheets

### Parquets/ProgressStatus/ProgressStatus_Output.xlsx
- Sheets (4): RawData, Diagnostics, Issues, Coverage
- `RawData` headers (25, header row 1): project_code, project_display, project_scope_key, line_name, line_name_source, section_label, source_file, source_sheet, configured_sheet, template_sheet, header_row_number, source_row_number, activity_raw, activity_norm, quantity_loa, quantity_estimated_or_total, quantity_primary, cumulative_last_month, plan_for_month, progress_for_month, today_progress, cumulative_progress, balance_progress, gangs_working, remarks
- `Diagnostics` headers (15, header row 1): Workbook, Project, Sheet, ConfiguredSheet, LineName, LineNameSource, TemplateSheet, TemplateApplied, TemplateChanges, FallbackNote, SectionsDetected, HeadersDetected, Rows, Status, Reason
- `Issues` headers (8, header row 1): Workbook, Project, Sheet, ConfiguredSheet, LineName, LineNameSource, Issue, Reason
- `Coverage` headers (10, header row 1): project_code, project_display, status, reason_code, reason, workbook, configured_sheet, resolved_sheet, rows, available_sheets

## Raw DPR Workbook Inventory (Sheet Names in Each Excel File)
- `Raw Data/DPRs/TA 310 - DPR - 2026-01-12.xlsx`: 13 sheets -> NKTL DPR PGCIL, Visual Chart NKTL, NKTL DPR, Sheet2, Sheet1, Visual Chart-Old, Location wise status, Survey Status, Soil investigation, Crossing Details, Line Scenario, Sheet3, Supply Status
- `Raw Data/DPRs/TA 325 - DPR - 2026-01-19.xlsx`: 21 sheets -> 101-276 FINAL 11-07-2025, 765KV TA 325 ANTL, Project Details, Visual chart Compiled, Soil Invst., Foundation-Compliled, FDN-VDRA, FDN-NVSR, Erection-Compiled, EREC-VDRA, EREC-NAV, Stringing Compiled, Striging-VDRA, String-Nav, Visual chart NAVSARI Revised, Crossing Vadodara, Crossing Navsari , Gangs, OPGW, Splicing, Add. Earthing
- `Raw Data/DPRs/TA 413 - DPR - 2026-04-14.xlsx`: 31 sheets -> TA413_PBNTL, Visual Chart, Final checking , Stretch readiness , Stringing, Erection, L2 Plan vs Actual, Survey summary, Detailed Survey , Check Survey , Tack Welding, Earthing, OPGW, Foundation, JUMPER Details, ROW Tracker, S-Curve FDN, S-Curve Erection, S-Curve Stringing, Accessories, Supply Status, Hardware Supply, Crossing status, Tower Abstract sheet, Statutory Proposals, Tower Abstarct , Location Summary, Foundation Daily Progress, Tower PRS abstract , Stretch readiness  (2), Area | NOTE: pandas read failed (ValueError); sheet names recovered via workbook.xml
- `Raw Data/DPRs/TA 414 - DPR - 2026-03-26.xlsx`: 21 sheets -> 765KV TA 414 VNTL, Project Details, VNTL - KEC, DPR Sheet, Sheet5, Sheet3, Sheet1, FDN, Erection , Erection Compiled , Sheet6, Sheet4, Sheet7, Stringing Compiled, String, Visual chart 1 , Crossing, Sheet2, Soil Inv., Survey Anx, Survey | NOTE: pandas read failed (ValueError); sheet names recovered via workbook.xml
- `Raw Data/DPRs/TA 416 - DPR - 2026-04-17.xlsx`: 12 sheets -> Project Details, MASTER SHEET, Erection Compiled , Stringing compiled, Progress., Visual chart., PENDING FDN DETAILS, PENDING ERE DETAILS, TSE-MANUAL-LT-11KV-33KV Details, Crossing Details, TS, TS (2)
- `Raw Data/DPRs/TA 418 - DPR - 2026-04-25.xlsx`: 12 sheets -> MASTER SHEET, Progress, Erection Compiled, Stringing Compiled, Visual Chart, ROW TRACKER, PENDING FDN DETAILS, TSE-MANUAL-LT-11KV-33KV Details, Crossing Details, Project Details, TS, TS (2)
- `Raw Data/DPRs/TA 419 - DPR - 2026-04-27.xlsx`: 39 sheets -> Const Revenue, DPR_TA419, Soil Inv, TOWER SUPPLY, RATE, Tack Welding, Ins Hosting, Stringing, L2, Supply Status, ROW, Tower Schedule, FDN, Latest Tower Schedule, Sheet8, Sheet7, Erection productivity, Foundation Daily Progress, Sheet9,  Summary, Sheet10, Visual Chart, Tower schedule_07.03.25, ERT, ERECTION COMPILED, Stringing Stretch Readiness, Project details, Supply, Stock Details, PGCIL, Detailed Survey, dont change, TOWER ABSTRACT, Pkg 2_Foundation, Foundation Daily Progresss, Erection Daily Progress, Statutory, Pkg 2_Erection, Pkg2_Stringing | NOTE: pandas read failed (ValueError); sheet names recovered via workbook.xml
- `Raw Data/DPRs/TA 421 - DPR - 2026-04-27.xlsx`: 14 sheets -> Project Details, Master Sheet, ProGress Sheet , Erection Compiled , Stringing Compiled , Erection Productivity , Visual Chart , Stubs and Towers , SuPPLY , X-ing Details , Tack Welding , Tower Accessories , Tower Tightening , Hardwares & Tower Accessories 
- `Raw Data/DPRs/TA 504 - DPR - 2026-04-23.xlsx`: 7 sheets -> DPR_TA-504, Erection compiled, Stringing, VC_RMTL, VC_Lilo, Project Details, Location Summary | NOTE: pandas read failed (ValueError); sheet names recovered via workbook.xml
- `Raw Data/DPRs/TA 505 - DPR - 2026-04-17.xlsx`: 15 sheets -> DPR-Summary, Sheet1, Project Details, Sheet5, Foundation, Balance Erc-RAJ, ERC ROW-MP, Erection Compiled, Stringing Compiled, Visual Chart, Erection Productivity, Earthing Status, Sheet4, Sheet3, Sheet2
- `Raw Data/DPRs/TA 506 - DPR - 2026-04-23.xlsx`: 6 sheets -> DPR_TA-506, Erection compiled, Stringing, Visual Chart, Project Details, Location Summary | NOTE: pandas read failed (ValueError); sheet names recovered via workbook.xml
- `Raw Data/DPRs/TA 509 - DPR - 2026-01-17.xlsx`: 12 sheets -> VC-765kV MKTL, SUMMARY, FOUNDATION, ERECTION, EARTHING, XING, SUPPLY, Sheet1, SURVEY SUMMARY, SURVEY STATUS , Erection productivity, TS  25-10-25 
- `Raw Data/DPRs/TA 512 - DPR - 2026-04-27.xlsx`: 25 sheets -> Summary S, Supply Status, Summary E, DPR Sheet , DS, CS, Hindrance Record, Earthing, Foundation, Erection Compiled, Tack Welding, Visual Chart, WIP, Statuary clearance, Stringing, OPGW, Accessories, Shield Wire Earthing, TS Gantry - AP47, Project Details, Sheet1, L3 Plan, Supply DPR , Power line Crossings , Hardware
- `Raw Data/DPRs/TA 513 - DPR - 2026-01-08.xlsx`: 27 sheets -> Concrete, Summary DPR, DPR S-F & S-P (2), DPR S-F & S-P, Survey , VC -SF, VC -SP, Ere S-P  , Land Schedule, Erection Compiled, Sheet5, Ere S-F , FDN S-F, Earthing S-P, Earthing S-F, FDN S-P, Supply, Proposal ., ROW S-F, Proposal, ROW S-P, TS SP, TS SF, Sheet4, Sheet2, Sheet1, Forest
- `Raw Data/DPRs/TA 601 - DPR - 2026-04-26.xlsx`: 11 sheets -> Visual Chart , Master Sheet, Progress Sheet , ROW, SUPPLY, Erection Compiled, X-ing Details , Visual Chart, Foundation & Erection, STRG Stretch , Stringing Compiled
- `Raw Data/DPRs/TB 401 - DPR - 2026-03-25.xlsx`: 45 sheets -> TB401 DPR....., Eng., Fdn (Pkg -G Section -I), TB401 DPR, Foundation (Pkg-F) , Earthing (PKG-F), Project Details, Progress, Foundation (Pkg-G), Earthing (PKG-G),  Benching Status-G, Revetment Pkg-G, Erection Compiled, Tack Welding - PKG-G, Stringing - PKG-G, OPGW PKG-G, Tower Accessories, Span Marker& Divertor, Avaition Light & Painting, Visual Chart PKG-G, Hindrance,  Benching Status-F, Earthing Pkg -G Ist section, Earthing balance, Tack Welding - PKG-F, Painting - PKG-F, Painting - PKG-G, Idling details from Aug, Crane Erection PKG-G, Erection Productivity, Eretion - PKG-F, Sheet1, JMC 27.9.23, Jumpering-G, Insulator Hoisting - F, Insulator Hoisting - G, Jumpering-F, Stringing - PKG-F, Stringing Productivity, OPGW PKG-F, Crossing, Visual Chart (Pkg -F.), Clear locs shared by client, Sheet3, Sheet2
- `Raw Data/DPRs/TB 408 - DPR - 2026-04-16.xlsx`: 22 sheets -> Progress Summary, Project Details, X-ing Status, L2 Schedule vs Actual, Hindrance Register (Row), Page Chart, Survey, Foundation, Earthing, Erection Productivity, Crane Erection Productivity, Erection Compiled, Tackwelding, Stringing, Stringing associated works, Stringing Productivity, OPGW, Accessories details, Visual chart, Visual chart Edit, Benching F, Incharge
- `Raw Data/DPRs/TB 501 - DPR - 2026-04-19.xlsx`: 14 sheets -> Detail Survey-220kV, Check Survey-220kV, Detail Survey-132kV, Check Survey-132kV, Summary, Statutory, Progress-132kV, Progress-220kV, Stringing-220kV, Stringing-132KV, VC-220kV, VC-132kV, Supply, Mail Body
- `Raw Data/DPRs/TB 507 - DPR - 2026-04-27.xlsx`: 33 sheets -> 765kV Summery, Check Survey Status, Check Survey, Sheet8, Sheet2, Sheet1, VC 51-96, VC, ROW, June, Check survey Loop out, Check survey Loop In, Foundation, Royalty, Sheet14, Sheet7, Project Details, Erection , Completion date, Erection Compiled , Earthing, Stringing Compiled , Soil Investigation, Sheet9, VC , Sheet6, VC 51-121, Sheet5, Sheet3, Tower Schedule, Visual Chart, L2 Vs Actual Progress Chart (2), Sheet4
- `Raw Data/DPRs/TB 507 [MAIN] - DPR - 2026-04-27.xlsx`: 36 sheets -> 400kV Summery, Check Survey , Check Survey Status, Check Survey, Sheet8, Sheet2, Project Details, Statutory Clearence,  Crossing Proposal, Sheet1, VC 51-96, VC, ROW, June, Foundation, Royalty, Sheet14, Sheet7, Erection , Completion date, Erection Compiled , Earthing, Final Check In And Tack Welding, Stringing Compiled , Soil Investigation, Sheet9, VC , Sheet6, VC 51-121, Sheet5, Sheet3, Tower Schedule, Visual Chart, L2 Vs Actual Progress Chart, L2 Vs Actual Progress Chart (2), Sheet4
- `Raw Data/DPRs/TB 605 - DPR - 2026-04-27.xlsx`: 72 sheets -> TB401 DPR....., Eng., Fdn (Pkg -G Section -I), Sheet2, Sheet3, Sheet4, Sheet5, ROW, Sheet13 (2), TB-605, Sheet16, Sheet14, Sheet12, Sheet13, Sheet15, TB401 DPR, Survey-Summary, Sheet6, Approved Survey, Approved CS J-S Line, Approved CS K-S Line, Foundation JS-Line -Jammu, Sheet10, Erection -Jammu, Vertical Chart -JS Line Jammu , Sheet17, Erection -SK, Sheet7, Sheet8, Foundation -LILO, Erection-Punjab, Vertical Chart -JS Line Punjab, Foundation -JS Line- Punjab, Sheet9, ROW Front vs Progress, Hindrance, Foundation -KS Line, Manpower, JS-Supply Stubs, JS Supply Towers, SK Supply Stubs, SK Supply Towers, Sheet11, Planned Vs Actual -FDN, Supply Status, Material Status-Jammu-Punjab, Material Status-Punjab, Earthing (PKG-F), Earthing (PKG-G), Revetment Pkg-G,  Benching Status-F,  Benching Status-G, Earthing Pkg -G Ist section, Earthing balance, Tack Welding - PKG-G, Tack Welding - PKG-F, Painting - PKG-G, Painting - PKG-F, Eretion - PKG-G, Eretion - PKG-F, Sheet1, JMC 27.9.23, Jumpering-G, Insulator Hoisting - F, Insulator Hoisting - G, Jumpering-F, Stringing - PKG-F, Stringing - PKG-G, OPGW PKG-G, OPGW PKG-F, Crossing, Clear locs shared by client
- `Raw Data/DPRs/TEST 000 - DPR - 2026-03-09.xlsx`: 14 sheets -> Detail Survey-220kV, Check Survey-220kV, Detail Survey-132kV, Check Survey-132kV, Summary, Statutory, Progress-132kV, Progress-220kV, Stringing-220kV, Stringing-132KV, VC-220kV, VC-132kV, Supply, Mail Body

## DPR_Config Mapping Used by New Pipelines
Columns shown: `Project Code`, `Stringing Sheet Names`, `Stringing Line Names`, `Status Sheet Names`, `Status Line Names`, `Stretch Readiness Sheet Names`, `Stretch Daily Stringing Sheet Names`, `Stretch Line Names`, `Stretch Manpower Expected`.

| Project Code | Stringing Sheet Names | Stringing Line Names | Status Sheet Names | Status Line Names | Stretch Readiness Sheet Names | Stretch Daily Stringing Sheet Names | Stretch Line Names | Stretch Manpower Expected |
|---|---|---|---|---|---|---|---|---|
| TB 605 |  |  |  |  |  |  |  |  |
| TA 414 | Stringing Compiled |  | VNTL - KEC |  |  | Stringing Compiled |  | Unknown |
| TA 504 | Stringing |  | DPR_TA-504 |  | Stringing | Stringing |  | Yes |
| TA 506 | Stringing |  | DPR_TA-506 |  | Stringing | Stringing |  | Yes |
| TB 507 | Stringing Compiled |  | 400kV Summery; 765kV Summery | 400kV; 765kV | Final Check In And Tack Welding | Stringing Compiled | MAIN | No |
| TA 512 |  |  |  |  |  |  |  |  |
| TA 413 | Stringing |  | TA413_PBNTL |  | Stretch readiness | Stringing |  | No |
| TA 419 | Stringing Stretch Readiness |  | DPR_TA419 |  | Stringing Stretch Readiness |  |  | Unknown |
| TA 418 | Stringing Compiled |  | MASTER SHEET |  | Stringing Compiled | Stringing Compiled |  | Yes |
| TA 421 | Stringing Compiled |  | Master Sheet |  | Stringing Compiled | Stringing Compiled |  | Yes |
| TA 416 | Stringing compiled |  | MASTER SHEET |  |  | Stringing compiled |  | No |
| TA 505 | Stringing Compiled |  | DPR-Summary |  | Stringing Compiled | Stringing Compiled |  | Yes |
| TB 408 | Stringing |  | Progress Summary |  |  | Stringing |  | No |
| TB 401 | Stringing - PKG-G |  |  |  |  |  |  |  |
| TA 325 | Stringing Compiled |  |  |  |  |  |  |  |
| TA 509 |  |  |  |  |  |  |  |  |
| TA 601 | Stringing Compiled |  |  |  |  |  |  |  |
| TB 501 | Stringing-132KV; Stringing-220kV | Stringing 132kV; Stringing 220kV | Summary |  |  | Stringing-220kV; Stringing-132KV | Stringing 220kV; Stringing 132kV | No |
| TA 310 |  |  |  |  |  |  |  |  |
| TA 513 |  |  |  |  |  |  |  |  |

## New Dataset Data Dictionary

### Stringing Manpower (`StringingSummary_Output.xlsx` -> `ManpowerProductivityFact`)
| Column | Meaning |
|---|---|
| `project_code` | Project code (e.g., TA 418, TB 507). |
| `project_display` | Friendly project label used in dashboard/grouping. |
| `project_scope_key` | Normalized project+line key for aggregations. |
| `line_name` | Transmission line variant name where available. |
| `date` | Work date for manpower/productivity observation. |
| `month` | Month bucket derived from date. |
| `gang_name` | Gang/contractor crew name. |
| `from_ap` | From location/AP for the span. |
| `to_ap` | To location/AP for the span. |
| `span_key` | Normalized span identifier (from-to composite). |
| `method` | Stringing method (TSE/Manual/other normalized values). |
| `section_readiness` | Readiness state from source sheet for that span/section. |
| `daily_km` | Daily productive km attributed to record. |
| `po_km` | Paying-out km value used in productivity calculations. |
| `manpower_gang_strength` | Gang strength/headcount captured from sheet. |
| `manpower_fitters` | No. of fitters captured from sheet. |
| `manpower_signal_type` | Type of manpower signal detected (columns/heuristic). |
| `manpower_status` | Derived status indicating manpower data presence/quality. |
| `expected_manpower` | Expected manpower availability config from DPR_Config. |
| `expected_match` | Whether observed manpower signal matched expectation. |
| `availability` | Availability bucket derived by pipeline. |
| `availability_reason` | Rule-based reason for chosen availability bucket. |

### Stretch Readiness (`StretchReadiness_Output.xlsx` -> `RawData`)
| Column | Meaning |
|---|---|
| `project_code` | Project code. |
| `project_display` | Friendly project label. |
| `project_scope_key` | Normalized project+line key. |
| `line_name` | Configured/resolved line name. |
| `line_name_source` | Where line_name came from (config/sheet inference). |
| `section_label` | Original section text from source row. |
| `source_file` | DPR workbook path used for row extraction. |
| `source_sheet` | Actual sheet name read. |
| `configured_sheet` | Sheet name configured in DPR_Config. |
| `template_sheet` | Template mapping sheet from DPR_Config workbook (if applied). |
| `report_date` | Report/DPR date resolved for row. |
| `header_row_number` | Detected header row index in source sheet. |
| `source_row_number` | Original row number in source sheet. |
| `stretch_identifier` | Derived unique stretch/section identifier. |
| `from_ap` | From AP/location for stretch. |
| `to_ap` | To AP/location for stretch. |
| `length_m_raw` | Raw length value in meters from source. |
| `length_km` | Normalized length in km. |
| `readiness_raw` | Raw readiness text from sheet. |
| `final_check_raw` | Raw final-checking text/status. |
| `tack_welding_raw` | Raw tack-welding text/status. |
| `balance_towers` | Raw/derived balance towers field. |
| `readiness_state` | Normalized readiness bucket (ready/partial/not_ready/unknown). |
| `remarks` | Free-text remarks from source row. |

### Progress Status (`ProgressStatus_Output.xlsx` -> `RawData`)
| Column | Meaning |
|---|---|
| `project_code` | Project code. |
| `project_display` | Friendly project label. |
| `project_scope_key` | Normalized project+line key. |
| `line_name` | Configured/resolved line name. |
| `line_name_source` | Where line_name came from. |
| `section_label` | Section/subgroup label in source block. |
| `source_file` | DPR workbook path used for row extraction. |
| `source_sheet` | Actual status/progress sheet read. |
| `configured_sheet` | Sheet configured in DPR_Config. |
| `template_sheet` | Template mapping sheet in DPR_Config (if applied). |
| `header_row_number` | Detected header row in source sheet. |
| `source_row_number` | Original row number from source sheet. |
| `activity_raw` | Raw activity name text. |
| `activity_norm` | Normalized activity key used for grouping. |
| `quantity_loa` | LOA quantity if present. |
| `quantity_estimated_or_total` | Estimated/total quantity if LOA not present. |
| `quantity_primary` | Primary quantity used by pipeline for progress math. |
| `cumulative_last_month` | Cumulative progress up to previous month. |
| `plan_for_month` | Planned progress for current month. |
| `progress_for_month` | Actual achieved progress for month. |
| `today_progress` | Today-specific progress value. |
| `cumulative_progress` | Current cumulative progress. |
| `balance_progress` | Remaining balance progress. |
| `gangs_working` | Working gangs count/details. |
| `remarks` | Free-text notes from source row. |

## 10-Row Samples for New Sheets

### Stringing manpower sample
Source: `Parquets/StringingSummary/StringingSummary_Output.xlsx` -> `ManpowerProductivityFact`
```csv
project_code,project_display,project_scope_key,line_name,date,month,gang_name,from_ap,to_ap,span_key,method,section_readiness,daily_km,po_km,manpower_gang_strength,manpower_fitters,manpower_signal_type,manpower_status,expected_manpower,expected_match,availability,availability_reason
TA 325,TA 325,ta325,,2024-01-08,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-09,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-10,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-11,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-12,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-13,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-14,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-15,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-16,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
TA 325,TA 325,ta325,,2024-01-17,2024-01-01,Dhervendra Electricals,144A/0,145/0,144A/0|145/0,Hotline,Ready,0.001068763440860215,0.39758,,,UNKNOWN,,,,NO_DATA,No manpower values found for this span/day.
```

### Stretch readiness sample
Source: `Parquets/StretchReadiness/StretchReadiness_Output.xlsx` -> `RawData`
```csv
project_code,project_display,project_scope_key,line_name,line_name_source,section_label,source_file,source_sheet,configured_sheet,template_sheet,report_date,header_row_number,source_row_number,stretch_identifier,from_ap,to_ap,length_m_raw,length_km,readiness_raw,final_check_raw,tack_welding_raw,balance_towers,readiness_state,remarks
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,11,Gantry - AP-1,,,150.014,0.150014,,,,,UNKNOWN,pending due to gantry not yet ready
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,12,AP-1 - AP-2A,,,290.21,0.29021,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,13,AP/2A - AP/3A,,,2836.992,2.836992,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,14,AP3A/0-AP3B/0,,,2691.467,2.691467,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,15,AP3B/0-AP4/0,,,1627.07,1.62707,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,16,AP-4 - AP-4A/0,,,709.163,0.709163,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,17,AP-4A/0 - AP-5,,,2681.611,2.681611,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,18,AP-5 - AP-5A/0,,,4418.389,4.418389,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,19,AP-5A/0 - AP-6,,,419.48,0.41948,,,,,UNKNOWN,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,Stretch readiness ,Stretch readiness,TA 413 Stretch,2026-04-14,9,20,AP-6 - AP-7,,,4901.236000000001,4.901236000000001,,,,,UNKNOWN,Completed
```

### Stretch manpower-audit sample
Source: `Parquets/StretchReadiness/StretchReadiness_Output.xlsx` -> `ManpowerAudit`
```csv
project_code,project_display,project_scope_key,line_name,line_name_source,source_file,source_sheet,configured_sheet,header_row_number,manpower_fields,readiness_fields,readiness_column_present,signal_type,non_empty_count,sample_values,expected_manpower,expected_match,status,reason
TA 413,TA 413,ta413,,filename,TA 413 - DPR - 2026-04-14.xlsx,Stringing,Stringing,0.0,,,False,ABSENT,0,,no,True,ABSENT,
TA 414,TA 414,ta414,,filename,TA 414 - DPR - 2026-03-26.xlsx,Stringing Compiled,Stringing Compiled,2.0,gang strength,Section Readiness,True,HEADER_ONLY,0,,unknown,True,HEADER_ONLY,
TA 416,TA 416,ta416,,filename,TA 416 - DPR - 2026-04-17.xlsx,Stringing compiled,Stringing compiled,2.0,,Section Readiness,True,ABSENT,0,,no,True,ABSENT,
TA 418,TA 418,ta418,,filename,TA 418 - DPR - 2026-04-25.xlsx,Stringing Compiled,Stringing Compiled,2.0,gang strength,Section Readiness,True,PRESENT_WITH_VALUES,15,73; 20; 58,yes,True,PRESENT_WITH_VALUES,
TA 419,TA 419,ta419,,,,,,,,,False,NO_SHEET_CONFIG,0,,unknown,True,NO_SHEET_CONFIG,No daily stringing sheet configured for manpower audit.
TA 421,TA 421,ta421,,filename,TA 421 - DPR - 2026-04-27.xlsx,Stringing Compiled ,Stringing Compiled,2.0,gang strength,Section Readiness,True,PRESENT_WITH_VALUES,51,35,yes,True,PRESENT_WITH_VALUES,
TA 504,TA 504,ta504,,filename,TA 504 - DPR - 2026-04-23.xlsx,Stringing,Stringing,2.0,gang strength,Section Readiness,True,PRESENT_WITH_VALUES,9,50,yes,True,PRESENT_WITH_VALUES,
TA 505,TA 505,ta505,,filename,TA 505 - DPR - 2026-04-17.xlsx,Stringing Compiled,Stringing Compiled,3.0,gang strength,Section Readiness,True,PRESENT_WITH_VALUES,10,30,yes,True,PRESENT_WITH_VALUES,
TA 506,TA 506,ta506,,filename,TA 506 - DPR - 2026-04-23.xlsx,Stringing,Stringing,1.0,gang strength,Section Readiness,True,PRESENT_WITH_VALUES,30,40; 39,yes,True,PRESENT_WITH_VALUES,
TB 408,TB 408,tb408,,filename,TB 408 - DPR - 2026-04-16.xlsx,Stringing,Stringing,3.0,,,False,ABSENT,0,,no,True,ABSENT,
```

### Progress data sample
Source: `Parquets/ProgressStatus/ProgressStatus_Output.xlsx` -> `RawData`
```csv
project_code,project_display,project_scope_key,line_name,line_name_source,section_label,source_file,source_sheet,configured_sheet,template_sheet,header_row_number,source_row_number,activity_raw,activity_norm,quantity_loa,quantity_estimated_or_total,quantity_primary,cumulative_last_month,plan_for_month,progress_for_month,today_progress,cumulative_progress,balance_progress,gangs_working,remarks
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,7,Route Alignment (Km),route_alignment,112.611,117.850851,117.850851,117.850851,,0.0,0.0,117.850851,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,8,Detailed Survey Completed(Km),detailed_survey,112.611,117.850851,117.850851,117.850851,,0.0,0.0,117.850851,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,9,Detailed Survey Submitted(Km),detailed_survey,112.611,117.850851,117.850851,117.850851,,0.0,0.0,117.850851,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,10,Detail survey approved (Kms),detail_survey_approved_kms,112.611,117.850851,117.850851,117.850851,,0.0,0.0,117.850851,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,13,Soil Investigation (Nos.),soil_investigation,29.0,29.0,29.0,26.0,,0.0,0.0,26.0,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,14,Check Survey approved(Km),check_survey,112.611,117.850851,117.850851,117.850851,,0.0,0.0,117.850851,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,15,Foundation Classification.,foundation,287.0,316.0,316.0,316.0,0.0,0.0,0.0,316.0,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,16,Excavation,excavation,287.0,316.0,316.0,316.0,0.0,0.0,0.0,316.0,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,17,Foundation (Nos.),foundation,287.0,316.0,316.0,316.0,0.0,0.0,0.0,316.0,0.0,,Completed
TA 413,TA 413,ta413,,filename,,TA 413 - DPR - 2026-04-14.xlsx,TA413_PBNTL,TA413_PBNTL,TA 413 Status,6,18,Earthing (Nos.),earthing,287.0,309.0,309.0,307.0,0.0,0.0,0.0,307.0,2.0,,
```

## Notes
- Some DPR workbooks have malformed XML (common in manually edited files). Sheet names were still recovered where possible via `xl/workbook.xml`.
- Many raw DPR tabs use merged/multi-row headers. Header rows above are best-effort detections for schema reference.
- The canonical normalized schemas consumed by dashboards are the compiled outputs under `Parquets/*_Output.xlsx`.