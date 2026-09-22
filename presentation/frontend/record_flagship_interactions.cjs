const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const ARTIFACTS_DIR = 'C:/Users/ADMIN/.gemini/antigravity/brain/1da2a9bb-bc6b-4531-a202-a5d44499c205/recordings';

if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function clickButtonWithText(page, searchText) {
  return page.evaluate((text) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const target = buttons.find(b => b.innerText && b.innerText.toLowerCase().includes(text.toLowerCase()));
    if (target) {
      target.click();
      return true;
    }
    return false;
  }, searchText);
}

(async () => {
  console.log('Launching headless Chrome...');
  const browser = await puppeteer.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-gpu',
      '--use-gl=swiftshader',
      '--window-size=1600,1000'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });

  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.toString()));

  console.log('Navigating to http://localhost:5173...');
  await page.goto('http://localhost:5173', { waitUntil: 'domcontentloaded', timeout: 15000 });
  await page.waitForSelector('canvas', { timeout: 15000 });
  console.log('Canvas ready, waiting for scene init...');
  await sleep(3000);

  // 1. Enter Scene 2: Manufacturing Line
  console.log('Switching to Scene 2 (Process Explorer)...');
  await page.keyboard.press('2');
  await sleep(2500);

  // Take overview screenshot
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '01_scene2_overview.png') });
  console.log('Saved 01_scene2_overview.png');

  // 2. Select Calendering stage (Station 05 for Warwick)
  console.log('Selecting Calendering stage via "Enter flagship"...');
  const clickedFlagship = await clickButtonWithText(page, 'Enter flagship');
  console.log('Clicked flagship:', clickedFlagship);
  await sleep(2000);

  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '02_warwick_calendering_explore.png') });
  console.log('Saved 02_warwick_calendering_explore.png');

  // Click "Let AI choose the calendering conditions"
  console.log('Clicking begin decision...');
  const clickedBegin = await clickButtonWithText(page, 'choose the calendering conditions');
  console.log('Clicked begin decision:', clickedBegin);
  await sleep(1500);

  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '03_warwick_ai_decision_proposal.png') });
  console.log('Saved 03_warwick_ai_decision_proposal.png');

  // Click "Run illustrated process response"
  console.log('Running illustrated process response for Warwick...');
  const clickedRun = await clickButtonWithText(page, 'Run illustrated process response');
  console.log('Clicked run process:', clickedRun);

  // Capture frames during choreography
  for (let i = 1; i <= 8; i++) {
    await sleep(400);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, `04_warwick_choreography_${i}.png`) });
    console.log(`Saved 04_warwick_choreography_${i}.png`);
  }

  // Result should now be revealed!
  await sleep(800);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '05_warwick_result_revealed.png') });
  console.log('Saved 05_warwick_result_revealed.png');

  // Click "Bridge to Measurement Station (Station 06)"
  console.log('Clicking Bridge to Measurement Station...');
  const clickedBridge = await clickButtonWithText(page, 'Bridge to Measurement Station');
  console.log('Clicked bridge button:', clickedBridge);
  if (clickedBridge) {
    await sleep(2500);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, '06_warwick_bridged_to_station06.png') });
    console.log('Saved 06_warwick_bridged_to_station06.png');
  }

  // Return to overview
  console.log('Testing reverse navigation to overview...');
  await page.evaluate(() => {
    const backBtn = document.querySelector('button[aria-label="Return to manufacturing overview"]');
    if (backBtn) backBtn.click();
  });
  await sleep(2000);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '07_warwick_returned_overview.png') });
  console.log('Saved 07_warwick_returned_overview.png');

  // 3. Switch Scenario to Drakopoulos Graphite
  console.log('Switching to Drakopoulos Graphite scenario...');
  await page.keyboard.press('s');
  await sleep(2500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '08_drakopoulos_overview.png') });
  console.log('Saved 08_drakopoulos_overview.png');

  // Enter flagship (Coating stage)
  console.log('Entering Drakopoulos flagship coating stage...');
  const clickedDrakFlagship = await clickButtonWithText(page, 'Enter flagship');
  console.log('Clicked Drak flagship:', clickedDrakFlagship);
  await sleep(2000);

  // Click "Let AI select a recipe"
  console.log('Clicking Let AI select a recipe...');
  const clickedDrakBegin = await clickButtonWithText(page, 'select a recipe');
  console.log('Clicked Drak begin:', clickedDrakBegin);
  await sleep(1500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '09_drakopoulos_recipe_proposal.png') });
  console.log('Saved 09_drakopoulos_recipe_proposal.png');

  // Click "Run illustrated process response"
  console.log('Running illustrated process response for Drakopoulos...');
  const clickedDrakRun = await clickButtonWithText(page, 'Run illustrated process response');
  console.log('Clicked Drak run:', clickedDrakRun);

  // Capture frames during choreography
  for (let i = 1; i <= 8; i++) {
    await sleep(400);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, `10_drakopoulos_choreography_${i}.png`) });
    console.log(`Saved 10_drakopoulos_choreography_${i}.png`);
  }

  // Wait for choreography completion and result reveal
  await sleep(800);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '11_drakopoulos_result_revealed.png') });
  console.log('Saved 11_drakopoulos_result_revealed.png');

  // Test repeat reverse navigation
  console.log('Testing repeat decision button...');
  const clickedRepeat = await clickButtonWithText(page, 'Repeat');
  console.log('Clicked repeat:', clickedRepeat);
  await sleep(1500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '12_drakopoulos_repeat_reversed.png') });
  console.log('Saved 12_drakopoulos_repeat_reversed.png');

  // Scene 4 check: Optimization Visualizer
  console.log('Navigating to Scene 4 (Optimization Visualizer)...');
  await page.keyboard.press('4');
  await sleep(2500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, '13_scene4_optimization_visualizer.png') });
  console.log('Saved 13_scene4_optimization_visualizer.png');

  await browser.close();
  console.log('All verification snapshots completed successfully.');
})();
