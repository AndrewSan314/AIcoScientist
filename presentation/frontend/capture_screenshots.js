import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SCREENSHOT_DIR = path.resolve('../screenshots');

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function run() {
  console.log('Launching Edge for Visual QA at ' + EDGE_PATH);
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });

  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.stack || err.message));
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });

  // 1. Discovery Setup State (Default)
  console.log('Capturing: desktop_discovery_setup.png');
  await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 2000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_discovery_setup.png') });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_discovery_lab.png') });

  // 2. Trigger Running State
  console.log('Capturing: desktop_discovery_running.png');
  const buttons = await page.$$('button');
  for (const b of buttons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && (text.includes('Execute 4-step campaign') || text.includes('Run discovery analysis') || text.includes('Open recorded analysis'))) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 600)); // capture during the 1.4s animation
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_discovery_running.png') });

  // 3. Results State (Wait for animation to complete)
  console.log('Capturing: desktop_discovery_results.png');
  await new Promise((r) => setTimeout(r, 1600));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_discovery_results.png') });

  // 3b. Click Predictive Distribution Tab
  console.log('Capturing: desktop_discovery_predictive_chart.png');
  const chartButtons = await page.$$('button');
  for (const b of chartButtons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.trim() === 'Predictive') {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_discovery_predictive_chart.png') });

  // 3c. Scroll down to capture the Heatmap and Score Decomposition
  console.log('Capturing: desktop_discovery_decision_matrix.png');
  await page.evaluate(() => window.scrollBy(0, 650));
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_discovery_decision_matrix.png') });
  await page.evaluate(() => window.scrollTo(0, 0));

  // 4. Evidence Workspace (Default Q1)
  console.log('Capturing: desktop_evidence_benchmarks.png');
  const buttons2 = await page.$$('button');
  for (const b of buttons2) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.trim().startsWith('Evidence')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_evidence_benchmarks.png') });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_evidence_benchmarks.png') });

  // 4b. Click Q2 (Sensitivity: MC12 vs MC32)
  console.log('Capturing: desktop_evidence_q2_sensitivity.png');
  const qButtons = await page.$$('button');
  for (const b of qButtons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Q2: Stress Robustness')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_evidence_q2_sensitivity.png') });

  // 4c. Click Q4 (A-Lab Calibration Coverage)
  console.log('Capturing: desktop_evidence_q4_calibration_xrd.png');
  for (const b of qButtons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Q4: Physical A-Lab Replay')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_evidence_q4_calibration_xrd.png') });

  // 4d. Toggle Rietveld Refinement on Q4
  console.log('Capturing: desktop_evidence_q4_calibration_rietveld.png');
  const modButtons = await page.$$('button');
  for (const b of modButtons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Rietveld Refinement')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_evidence_q4_calibration_rietveld.png') });

  // 5. How It Works Workspace
  console.log('Capturing: desktop_how_it_works.png');
  const buttons3 = await page.$$('button');
  for (const b of buttons3) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.trim().startsWith('How it works')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_how_it_works.png') });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_research_system.png') });

  // 6. Presenter Mode
  console.log('Capturing: desktop_presenter_mode.png');
  await page.keyboard.press('p');
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_presenter_mode.png') });
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_presenter_mode.png') });

  await browser.close();
  console.log('All Visual QA screenshots successfully captured in presentation/screenshots/');
}

run().catch((err) => {
  console.error('Error during Visual QA:', err);
  process.exit(1);
});
