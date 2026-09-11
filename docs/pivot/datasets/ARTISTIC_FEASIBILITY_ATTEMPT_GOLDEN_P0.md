# ARTISTIC feasibility attempt: golden-p0-thickness-full

Classification: `OUT_OF_PUBLISHED_DOMAIN / COMPUTATIONALLY_INFEASIBLE`; termination: user-requested stop. The local immutable attempt manifest and logs remain at `outputs/artistic_runs/golden-p0-thickness-full/` (`workspace/log.lammps`, `workspace/slurry.log`); their SHA-256 hashes are in the manifest.

| Item | Recorded value |
| --- | --- |
| Legacy rendered recipe | `dry_mass_mg=1.0` (upstream renders `dry_mass=1`, then `/1E6` g = 1 ug); CBD diameter 0.1 um; CBD nanoporosity 0.5; AM/CBD 0.9/0.1; solid 0.5; AM diameter 5 um |
| Published-domain violations | electrode mass 1.0 ug vs 0.1–0.2 ug; CBD diameter 0.1 um vs 0.7–1.5 um |
| Source-count output | 2,957 AM + 212,206,594 CBD = 212,209,551 total particles |
| Last completed creation | batch 1 of 20: 10,610,329 particles; batch 2 started but did not complete |
| Peak working set | 17,696,182,272 bytes |
| Elapsed wall / last observed CPU | 15,350.35174 s / 14,373.296875 s |
| LAMMPS | 22 Jul 2025 Update 4 |

This is not a normalized `BatteryProcessRun`, not a golden result, and must not enter DOE or surrogate training.
