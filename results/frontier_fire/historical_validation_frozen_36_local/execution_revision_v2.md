# Frozen-36 execution revision 2

Date: 2026-08-05  
Reason: deterministic source-archive incompleteness during preparation  
Affected incident: `NV-HTF-500736_Mahogany` (training split)

## Failure

The initial execution stopped at the metric-corpus preparation stage. HRRR
analysis groups were complete for 148 of 172 requested hours. For every hour
on 2 July 2020, UGRD, VGRD, TMP, and RH were present and PRATE was absent. The
loader discarded each incomplete hour and reported 0.860 coverage against the
frozen 0.950 minimum. One unchanged retry reproduced the failure.

## Revision

`fetch_hrrr_analysis` now performs an explicitly recorded, same-model repair
only when incomplete analysis groups would otherwise fail the existing
coverage rule:

1. reread an incomplete group field by field;
2. retain every available analysis field;
3. partition absent fields into contiguous hourly gaps;
4. select the newest preceding six-hour HRRR forecast cycle covering a gap;
5. fill only the absent field and cache that source tile; and
6. retain raw analysis coverage separately from final usable-weather coverage.

Mahogany uses only forecast PRATE from the 2020-07-01 18:00 UTC cycle, leads
F06 through F29. The resulting forcing contains 172 hourly samples, reports
148 direct-analysis hours, lists all 24 incomplete analysis timestamps, and
reports no unresolved weather hour.

The frozen incident membership, chronological split, perimeter observations,
parameter candidates, objectives, and development/test rules were not changed.

## Fingerprints

- Superseded tree: `019640719b63313e7ac66566864c7e7144e97a2dda3d632ce96a7b36030a8a93`
- Active revision-2 tree: `9d8f068e2c459ef14cbbbe140ba0ee1e2e1e2f864fb2eafa8b87da7fdee00637`
- Active manifest: `execution_source_manifest_v2_hrrr_gap_repair.json`

The superseded manifest is preserved. Among its recorded files, verification
identifies only `src/aeolus/data/hrrr.py` as changed. Revision 2 additionally
fingerprints `src/aeolus/data/gofer.py` and
`src/aeolus/data/progression.py`; these independent background-research
adapters are outside the frozen benchmark command/import closure. Verification
against revision 2 passes.

## Verification

- Focused HRRR, weather-window, forcing-validity, and frozen-contract tests:
  13 passed; the complete repository suite then passed 134 tests.
- Real archive repair: 24 partial analysis tiles and one forecast PRATE tile
  materialized.
- Real Mahogany bundle: `item.json` present; raw coverage 0.860465; usable
  coverage 1.0; 172 background samples overlaid; zero unresolved hours.
- Resumable preparation continued from 24 to 26 completed incidents before
  the benchmark controller was reattached.
