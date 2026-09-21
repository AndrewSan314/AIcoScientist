import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const BROWSER_PATH = process.env.SCREENSHOT_BROWSER || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SCREENSHOT_DIR = path.resolve('../screenshots/exhibition');
const ROOT_SCREENSHOT_DIR = path.resolve('../../screenshots');
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}
if (!fs.existsSync(ROOT_SCREENSHOT_DIR)) {
  fs.mkdirSync(ROOT_SCREENSHOT_DIR, { recursive: true });
}

async function saveScreenshot(page, filename) {
  const p1 = path.join(SCREENSHOT_DIR, filename);
  const p2 = path.join(ROOT_SCREENSHOT_DIR, filename);
  await page.screenshot({ path: p1 });
  await page.screenshot({ path: p2 });
  console.log(`[Captured] ${filename} -> ${p1}`);
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
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle0', timeout: 20000 });
    await new Promise((r) => setTimeout(r, 2000)); // Allow Three.js persistent canvas to render initial frames
    await saveScreenshot(page, '01_initial_hero.png');

    // 2. Manufacturing Line Overview (Scene 2)
    console.log('Navigating to Scene 2: Connected Production Line...');
    // Click Scene 2 button in SceneNavigator
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('nav button'));
      const scene2 = btns.find(b => b.textContent?.includes('Production Line') || b.textContent?.includes('Line'));
      if (scene2) scene2.click();
    });
    await new Promise((r) => setTimeout(r, 1800));
    await saveScreenshot(page, '02_manufacturing_overview.png');

    // 3. Focus on Coating Equipment in Scene 2
    console.log('Selecting Coating stage...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const coatingBtn = btns.find(b => b.textContent?.includes('Coating'));
      if (coatingBtn) coatingBtn.click();
    });
    await new Promise((r) => setTimeout(r, 1500));
    await saveScreenshot(page, '03_coating_equipment.png');

    // 4. Focus on Calendering Equipment in Scene 2
    console.log('Selecting Calendering stage...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const calenderBtn = btns.find(b => b.textContent?.includes('Calendering'));
      if (calenderBtn) calenderBtn.click();
    });
    await new Promise((r) => setTimeout(r, 1500));
    await saveScreenshot(page, '04_calendering_equipment.png');

    // 5. Scene 3: Inside the Electrode (Microstructure before morph: 0% compression)
    console.log('Navigating to Scene 3: Inside the Electrode...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('nav button'));
      const scene3 = btns.find(b => b.textContent?.includes('Inside the Electrode') || b.textContent?.includes('Microstructure'));
      if (scene3) scene3.click();
    });
    await new Promise((r) => setTimeout(r, 1800));

    // Helper for React controlled range input
    const setSliderValue = async (targetVal) => {
      await page.evaluate((v) => {
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
        }
      }, targetVal);
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
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('nav button'));
      const scene4 = btns.find(b => b.textContent?.includes('AI Optimization') || b.textContent?.includes('AI Studio'));
      if (scene4) scene4.click();
    });
    await new Promise((r) => setTimeout(r, 1800));
    await saveScreenshot(page, '08_warwick_optimization_studio.png');

    // 9. Advance Replay Step to Step 4 (Optimal Rediscovery EXP_03)
    console.log('Advancing Replay to Step 4 (Optimal condition EXP_03)...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const step4 = btns.find(b => b.textContent?.trim() === 'Step 4');
      if (step4) step4.click();
    });
    await new Promise((r) => setTimeout(r, 1200));

    // Inspect EXP_03 candidate
    await page.evaluate(() => {
      const allDivs = Array.from(document.querySelectorAll('div'));
      const exp03Card = allDivs.find(d => d.textContent?.includes('EXP_03') && d.classList.contains('cursor-pointer'));
      if (exp03Card) exp03Card.click();
    });
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '09_warwick_optimization_result.png');

    // 10. Scene 5: Scientific Evidence & Benchmark Telemetry
    console.log('Navigating to Scene 5: Scientific Evidence...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('nav button'));
      const scene5 = btns.find(b => b.textContent?.includes('Scientific Evidence') || b.textContent?.includes('Evidence'));
      if (scene5) scene5.click();
    });
    await new Promise((r) => setTimeout(r, 1800));
    await saveScreenshot(page, '10_scientific_evidence.png');

    // 11. Open Scientific Provenance & Limitations Drawer
    console.log('Opening Scientific Provenance Drawer...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const provBtn = btns.find(b => b.textContent?.includes('Scientific Provenance') || b.textContent?.includes('Known Limitations'));
      if (provBtn) provBtn.click();
    });
    await new Promise((r) => setTimeout(r, 1000));
    await saveScreenshot(page, '11_scientific_provenance_drawer.png');

    // Close Drawer
    await page.evaluate(() => {
      const closeBtn = Array.from(document.querySelectorAll('button')).find(b => b.textContent?.includes('Close'));
      if (closeBtn) closeBtn.click();
    });
    await new Promise((r) => setTimeout(r, 600));

    // 12. Switch to Drakopoulos Scenario & Navigate to Optimization
    console.log('Switching to Drakopoulos Graphite Anode Scenario...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('header button'));
      const drakBtn = btns.find(b => b.textContent?.includes('Drakopoulos'));
      if (drakBtn) drakBtn.click();
    });
    await new Promise((r) => setTimeout(r, 1000));

    // Go to Scene 4 in Drakopoulos
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('nav button'));
      const scene4 = btns.find(b => b.textContent?.includes('AI Optimization') || b.textContent?.includes('AI Studio'));
      if (scene4) scene4.click();
    });
    await new Promise((r) => setTimeout(r, 1200));

    // Select Step 4 in Drakopoulos
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const step4 = btns.find(b => b.textContent?.trim() === 'Step 4');
      if (step4) step4.click();
    });
    await new Promise((r) => setTimeout(r, 1200));

    // Inspect Recipe-01 candidate
    await page.evaluate(() => {
      const allDivs = Array.from(document.querySelectorAll('div'));
      const recipe01Card = allDivs.find(d => d.textContent?.includes('Recipe-01') && d.classList.contains('cursor-pointer'));
      if (recipe01Card) recipe01Card.click();
    });
    await new Promise((r) => setTimeout(r, 800));
    await saveScreenshot(page, '12_drakopoulos_optimization_studio.png');

    console.log('=== Automated Exhibition Visual QA Completed Successfully! ===');
    console.log('Total Screenshots Captured: 12');
    console.log('Total Critical Errors Encountered:', errors.length);
    if (errors.length > 0) {
      console.warn('Encountered page warnings/errors:', errors);
    }
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
