import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const SCREENSHOT_DIR = path.resolve('../screenshots');

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

  // 1. Desktop 1920x1080 Discovery Lab
  console.log('Capturing: desktop_1920_discovery_lab.png');
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto('http://localhost:8501/', { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 2000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_discovery_lab.png') });

  // 2. Desktop 1366x768 Discovery Lab
  console.log('Capturing: desktop_1366_discovery_lab.png');
  await page.setViewport({ width: 1366, height: 768, deviceScaleFactor: 1 });
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1366_discovery_lab.png') });

  // 3. Mobile 375x812 Discovery Lab
  console.log('Capturing: mobile_375_discovery_lab.png');
  await page.setViewport({ width: 375, height: 812, isMobile: true, deviceScaleFactor: 2 });
  await new Promise((r) => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'mobile_375_discovery_lab.png') });

  // Reset to 1920x1080
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto('http://localhost:8501/', { waitUntil: 'networkidle0', timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));

  // 4. Switch to Predictive Distributions Chart
  console.log('Capturing: desktop_1920_predictive_dist.png');
  const buttons = await page.$$('button');
  for (const b of buttons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Predictive Distributions')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_predictive_dist.png') });

  // 5. Switch to Evidence & Benchmarks Workspace
  console.log('Capturing: desktop_1920_evidence_benchmarks.png');
  for (const b of buttons) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Evidence & Benchmarks')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_evidence_benchmarks.png') });

  // 6. Switch to Research System Workspace
  console.log('Capturing: desktop_1920_research_system.png');
  const buttons2 = await page.$$('button');
  for (const b of buttons2) {
    const text = await page.evaluate((el) => el.textContent, b);
    if (text && text.includes('Research System')) {
      await b.click();
      break;
    }
  }
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_research_system.png') });

  // 7. Presenter Mode
  console.log('Capturing: desktop_1920_presenter_mode.png');
  await page.keyboard.press('p');
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_1920_presenter_mode.png') });

  await browser.close();
  console.log('All Visual QA screenshots successfully written to presentation/screenshots/');
}

run().catch((err) => {
  console.error('Error during Visual QA:', err);
  process.exit(1);
});
