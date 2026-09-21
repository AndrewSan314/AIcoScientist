# Process-native interaction audit

## Baseline

Before this overhaul, every manufacturing-stage interaction followed the same pattern: select a stage, move the camera, show a static parameter panel, and optionally open AI Studio. The 3D machines did not expose mechanism state, the AI decision did not occur inside the machine, and the historical result was visible outside a process run.

| Scenario | Stage | Previous response | Overhauled response |
| --- | --- | --- | --- |
| Warwick | Slurry preparation | Camera + text | Dosing exploded view with selectable material roles |
| Warwick | Pilot coating | Camera + text | Backing-roll, die-lip, and web inspection |
| Warwick | Calendering | Camera + text + route to Studio | In-place replay: controls → roller/web response → source-backed reveal |
| Warwick | Cell assembly | Camera + text | Electrode and assembly probe choreography |
| Warwick | Rate cycling | Camera + text | Measurement station plus latest revealed replay value |
| Drakopoulos | Formulation | Camera + text | Dosing exploded view with selectable material roles |
| Drakopoulos | Mixing | Camera + text | Vessel cutaway and agitator motion |
| Drakopoulos | Coating | Camera + text + route to Studio | Joint recipe replay: coating/drying/calendering controls → process response → source-backed reveal |
| Drakopoulos | Drying | Camera + text | Oven cutaway and qualitative IR-zone response |
| Drakopoulos | Calendering | Camera + text | Roller-nip cutaway and web response |
| Drakopoulos | Characterization | Camera + text | Measurement station plus latest revealed replay value |

## Scientific-integrity constraints

- Historical targets remain hidden until the illustrated process run completes.
- AI predictions are labelled predictions and remain distinct from historical measurements.
- Controls, recipe IDs, target values, units, and best-so-far values come from the replay dataset.
- Particle motion, cutaways, thermal color, compression bridges, and the measurement marker are explicitly illustrative; no CFD, thermal, pressure, pore, or electrochemical curve is inferred.
- `App` owns the shared scenario, selected stage, replay step, selected candidate, and process phase. Scene 2 and AI Studio read the same state.

## Acceptance path

1. Select Warwick and enter Calendering.
2. Ask AI for the next recorded condition; verify the historical target remains absent.
3. Run the illustrated mechanism; verify the result appears only when the run finishes.
4. Open AI Studio and return; verify recipe, replay step, and result state are preserved.
5. Repeat for Drakopoulos Coating, including coating speed, gap, dryer temperature, and calendering status.
