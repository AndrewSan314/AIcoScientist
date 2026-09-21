import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const BROWSER_PATH = process.env.SCREENSHOT_BROWSER || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SCREENSHOT_DIR = path.resolve(import.meta.dirname, '../screenshots/exhibition-final');
const ROOT_SCREENSHOT_DIR = path.resolve(import.meta.dirname, '../../screenshots');
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}
if (!fs.existsSync(ROOT_SCREENSHOT_DIR)) {
  fs.mkdirSync(ROOT_SCREENSHOT_DIR, { recursive: true });
}
async function writeWithRetry(targetPath, buffer, retries = 5) {
  for (let i = 0; i < retries; i++) {
    try {
      fs.writeFileSync(targetPath, buffer);
      return;
    } catch (e) {
      if (i === retries - 1) throw e;
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

async function saveScreenshot(page, filename) {
  const qa = await page.evaluate(() => window.__exhibitionQA);
  if (!qa?.modelVisible || qa.renderer.calls < 1 || qa.renderer.triangles < 1) {
    throw new Error(`WebGL assertion failed before ${filename}: ${JSON.stringify(qa)}`);
  }
  const p1 = path.join(SCREENSHOT_DIR, filename);
  const image = await page.screenshot();
  await writeWithRetry(p1, image);
  if (fs.existsSync(ROOT_SCREENSHOT_DIR)) {
    await writeWithRetry(path.join(ROOT_SCREENSHOT_DIR, filename), image);
  }
  console.log(`[Captured] ${filename} -> ${p1}`);
}

async function clickRequired(page, selector, labels, description) {
  const clicked = await page.evaluate(({ selector, labels }) => {
    const element = Array.from(document.querySelectorAll(selector))
      .find((node) => labels.some((label) => node.textContent?.includes(label)));
    if (!element) return false;
    element.click();
    return true;
  }, { selector, labels });
  if (!clicked) throw new Error(`Missing required interaction: ${description}`);
}

async function assertScene(page, scene, expectedText) {
  await page.waitForFunction((value) => window.__exhibitionQA?.scene === value, { timeout: 5000 }, scene);
  const state = await page.evaluate((text) => ({
    textPresent: document.body.innerText.includes(text),
    canvas: document.querySelector('canvas')?.getBoundingClientRect().toJSON(),
    qa: window.__exhibitionQA,
  }), expectedText);
  if (!state.textPresent || !state.canvas?.width || !state.canvas?.height || !state.qa?.modelVisible) {
    throw new Error(`Scene ${scene} assertion failed: ${JSON.stringify(state)}`);
  }
}

async function run() {
  console.log('=== Starting AIcoScientist Exhibition Visual QA & Automated Verification ===');
  console.log('Target Browser:', BROWSER_PATH);
  console.log('Base URL:', BASE_URL);

  const browser = await puppeteer.launch({
    executablePath: BROWSER_PATH,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--window-size=1920,1080'
    ]
  });

  const page = await browser.newPage();
  const errors = [];
  page.on('console', msg => {
    const txt = msg.text();
    if (msg.type() === 'error' && !txt.includes('favicon')) {
      errors.push(`[Console Error] ${txt}`);
      console.error('PAGE ERROR:', txt);
    }
  });
  page.on('pageerror', err => {
    errors.push(`[Page Error] ${err.message}`);
    console.error('PAGE CRASH:', err.message);
  });

  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });

  try {
    // 1. Initial Hero Scene (Scene 1)
    console.log('Navigating to exhibition root...');
    await page.goto(`${BASE_URL}/`, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await new Promise((r) => setTimeout(r, 2000)); // Allow Three.js persistent canvas to render initial frames
    await assertScene(page, 1, 'Battery Manufacturing · Scientific Exhibition');
    await saveScreenshot(page, '01_initial_hero.png');

    // 2. Manufacturing Line Overview (Scene 2)
    console.log('Navigating to Scene 2: Connected Production Line...');
    // Click Scene 2 button in SceneNavigator
    await clickRequired(page, 'nav button', ['Production Line'], 'Scene 2 navigation');
    await new Promise((r) => setTimeout(r, 1800));
    await assertScene(page, 2, 'MANUFACTURING LINE STAGES');
    await saveScreenshot(page, '02_manufacturing_overview.png');

    // 3. Focus on Coating Equipment in Scene 2
    console.log('Selecting Coating stage...');
    await clickRequired(page, 'button', ['Coating'], 'coating stage');
    await new Promise((r) => setTimeout(r, 1500));
    await assertScene(page, 2, 'Slot-die application');
    await saveScreenshot(page, '03_coating_equipment.png');

    // 4. Focus on Calendering Equipment in Scene 2
    console.log('Selecting Calendering stage...');
    await clickRequired(page, 'button', ['Calendering'], 'calendering stage');
    await new Promise((r) => setTimeout(r, 1500));
    await assertScene(page, 2, 'Roll temperature');
    await saveScreenshot(page, '04_calendering_equipment.png');

    // 4b. Test Cinematic Video Tour with live telemetry card
    console.log('Activating Cinematic Video Tour in Scene 2...');
    await clickRequired(page, 'button', ['Video Tour'], 'start video tour');
    await new Promise((r) => setTimeout(r, 1500));
    await saveScreenshot(page, '04b_cinematic_video_tour.png');

    // 4c. Test Clean Screen mode (Key 'H')
    console.log('Toggling Clean Screen mode (Key H)...');
    await page.keyboard.press('KeyH');
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '04c_clean_screen_recording.png');

    // Exit Clean Screen mode
    await page.keyboard.press('KeyH');
    await new Promise((r) => setTimeout(r, 600));

    // Exit Video Tour
    await clickRequired(page, 'button', ['Exit', 'Stop Tour'], 'exit video tour');
    await new Promise((r) => setTimeout(r, 600));

    // 5. Scene 3: Inside the Electrode (Microstructure before morph: 0% compression)
    console.log('Navigating to Scene 3: Inside the Electrode...');
    await clickRequired(page, 'nav button', ['Inside the Electrode'], 'Scene 3 navigation');
    await new Promise((r) => setTimeout(r, 350));
    await saveScreenshot(page, 'transition_calender_to_micro_035.png');
    await new Promise((r) => setTimeout(r, 550));
    await saveScreenshot(page, 'transition_calender_to_micro_090.png');
    await new Promise((r) => setTimeout(r, 1800));
    await assertScene(page, 3, 'ILLUSTRATIVE ELECTRODE CUTAWAY');

    // Helper for React controlled range input
    const setSliderValue = async (targetVal) => {
      const changed = await page.evaluate((v) => {
        const slider = document.querySelector('input[type="range"]');
        if (slider) {
          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
          if (setter) {
            setter.call(slider, String(v));
          } else {
            slider.value = String(v);
          }
          slider.dispatchEvent(new Event('input', { bubbles: true }));
          slider.dispatchEvent(new Event('change', { bubbles: true }));
          return true;
        }
        return false;
      }, targetVal);
      if (!changed) throw new Error(`Missing compaction slider for ${targetVal}`);
    };

    // Reset compression to 0%
    await setSliderValue(0);
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '05_electrode_microstructure_before_morph.png');

    // 6. Microstructure during morph (50% compression)
    console.log('Morphing microstructure to 50%...');
    await setSliderValue(0.5);
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '06_electrode_microstructure_during_morph.png');

    // 7. Microstructure after morph (100% compression)
    console.log('Morphing microstructure to 100%...');
    await setSliderValue(1.0);
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '07_electrode_microstructure_after_morph.png');

    // 8. Scene 4: AI Optimization Studio (Warwick NMC622)
    console.log('Navigating to Scene 4: AI Optimization Studio...');
    await clickRequired(page, 'nav button', ['AI Optimization'], 'Scene 4 navigation');
    await new Promise((r) => setTimeout(r, 450));
    await saveScreenshot(page, 'transition_micro_to_data_045.png');
    await new Promise((r) => setTimeout(r, 1800));
    await assertScene(page, 4, 'Candidate Search Space');
    await saveScreenshot(page, '08_warwick_optimization_studio.png');

    // 8b. Toggle 3D Turntable Orbit in Scene 4
    console.log('Toggling 3D Turntable Orbit in Scene 4...');
    await clickRequired(page, 'button', ['3D Orbit'], 'toggle 3D orbit');
    await new Promise((r) => setTimeout(r, 1200));
    await saveScreenshot(page, '08b_optimization_gp_manifold_orbit.png');

    // 9. Advance source-backed Warwick seed-11 replay to Step 5.
    console.log('Advancing Warwick replay to source-backed Step 5...');
    await clickRequired(page, 'button', ['Step 5'], 'Warwick replay Step 5');
    await new Promise((r) => setTimeout(r, 1200));

    await clickRequired(page, 'div.cursor-pointer', ['EXP_03'], 'EXP_03 candidate');
    await new Promise((r) => setTimeout(r, 800));
    await assertScene(page, 4, 'EXP_03');
    await saveScreenshot(page, '09_warwick_optimization_result.png');

    // 10. Scene 5: Scientific Evidence & Benchmark Telemetry
    console.log('Navigating to Scene 5: Scientific Evidence...');
    await clickRequired(page, 'nav button', ['Scientific Evidence'], 'Scene 5 navigation');
    await new Promise((r) => setTimeout(r, 1800));
    await assertScene(page, 5, 'Rigorous Methodology');
    await saveScreenshot(page, '10_scientific_evidence.png');

    // 11. Open Scientific Provenance & Limitations Drawer
    console.log('Opening Scientific Provenance Drawer...');
    await clickRequired(page, 'button', ['Scientific Provenance', 'Known Limitations'], 'provenance drawer');
    await new Promise((r) => setTimeout(r, 1000));
    await saveScreenshot(page, '11_scientific_provenance_drawer.png');

    // Close Drawer
    await clickRequired(page, 'button', ['Close'], 'close provenance drawer');
    await new Promise((r) => setTimeout(r, 600));

    // 12. Switch to Drakopoulos Scenario & Navigate to Optimization
    console.log('Switching to Drakopoulos Graphite Anode Scenario...');
    await clickRequired(page, 'header button', ['Drakopoulos'], 'Drakopoulos scenario');
    await new Promise((r) => setTimeout(r, 1000));

    // Go to Scene 4 in Drakopoulos
    await clickRequired(page, 'nav button', ['AI Optimization'], 'Drakopoulos Scene 4');
    await new Promise((r) => setTimeout(r, 1200));

    // Select Step 4 in Drakopoulos
    await clickRequired(page, 'button', ['Step 4'], 'Drakopoulos replay Step 4');
    await new Promise((r) => setTimeout(r, 1200));

    // Inspect Recipe-01 candidate
    await clickRequired(page, 'div.cursor-pointer', ['Recipe c1c280'], 'Drakopoulos best recipe');
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '12_drakopoulos_optimization_studio.png');

    console.log('=== Automated Exhibition Visual QA Completed Successfully! ===');
    console.log('Total Screenshots Captured: 12');
    console.log('Total Critical Errors Encountered:', errors.length);
    if (errors.length > 0) throw new Error(`Runtime errors: ${errors.join('\n')}`);

    const performance = await page.evaluate(() => ({
      ...window.__exhibitionQA,
      userAgent: navigator.userAgent,
      platform: navigator.platform,
    }));
    const sorted = [...performance.frameMs].sort((a, b) => a - b);
    performance.measurement = {
      samples: sorted.length,
      medianFrameMs: sorted[Math.floor(sorted.length * .5)],
      p95FrameMs: sorted[Math.floor(sorted.length * .95)],
      medianFps: 1000 / sorted[Math.floor(sorted.length * .5)],
      conditions: '1920x1080 viewport, deviceScaleFactor 1, headless Edge, WebGL enabled',
    };
    fs.writeFileSync(path.resolve(import.meta.dirname, '../../docs/exhibition/PERFORMANCE_QA.json'), JSON.stringify(performance, null, 2));
  } catch (err) {
    console.error('Visual QA Failure:', err);
    throw err;
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
