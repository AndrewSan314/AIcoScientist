import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { DRAKOPOULOS_SCENARIO, WARWICK_SCENARIO } from '../../data/exhibitionData';
import { ProcessDecisionExperience } from './ProcessDecisionExperience';

const render = (scenario: typeof DRAKOPOULOS_SCENARIO, phase: 'AI_DECISION' | 'RESULT_REVEAL', replayStep: number) => renderToStaticMarkup(
  <ProcessDecisionExperience
    scenario={scenario}
    phase={phase}
    revealedSteps={replayStep}
    onBeginDecision={() => undefined}
    onRunProcess={() => undefined}
    onNextDecision={() => undefined}
    onRepeatDecision={() => undefined}
    onOpenAdvanced={() => undefined}
  />,
);

describe('ProcessDecisionExperience reveal boundary', () => {
  it('does not render an unrevealed historical target', () => {
    const html = render(DRAKOPOULOS_SCENARIO, 'AI_DECISION', 0);
    expect(html).toContain('289.97');
    expect(html).not.toContain('263.89');
    expect(html).not.toContain('Historical measurement revealed');
  });

  it('reveals only the exact completed replay record', () => {
    const html = render(DRAKOPOULOS_SCENARIO, 'RESULT_REVEAL', 1);
    expect(html).toContain('Recipe d36021');
    expect(html).toContain('263.89');
    expect(html).toContain('Historical measurement revealed');
  });

  it('keeps Warwick values isolated from the graphite experience', () => {
    const html = renderToStaticMarkup(
      <ProcessDecisionExperience
        scenario={WARWICK_SCENARIO}
        phase="RESULT_REVEAL"
        revealedSteps={1}
        onBeginDecision={() => undefined}
        onRunProcess={() => undefined}
        onNextDecision={() => undefined}
        onRepeatDecision={() => undefined}
        onOpenAdvanced={() => undefined}
      />,
    );
    expect(html).toContain('0.6876');
    expect(html).not.toContain('263.89');
  });

  it('renders bridge to measurement station button when revealed', () => {
    const html = renderToStaticMarkup(
      <ProcessDecisionExperience
        scenario={WARWICK_SCENARIO}
        phase="RESULT_REVEAL"
        revealedSteps={1}
        onBeginDecision={() => undefined}
        onRunProcess={() => undefined}
        onNextDecision={() => undefined}
        onRepeatDecision={() => undefined}
        onOpenAdvanced={() => undefined}
        onInspectMeasurement={() => undefined}
      />,
    );
    expect(html).toContain('Bridge to Measurement Station (Station 06)');
  });
});

