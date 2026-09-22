const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const FRAMES_DIR = path.join(__dirname, 'temp_video_frames');
if (fs.existsSync(FRAMES_DIR)) {
  fs.rmSync(FRAMES_DIR, { recursive: true, force: true });
}
fs.mkdirSync(FRAMES_DIR, { recursive: true });

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
  console.log('1. Launching headless Chrome with unthrottled rendering flags...');
  const browser = await puppeteer.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      '--window-size=1600,900'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900 });
  await page.bringToFront();

  console.log('2. Navigating to http://localhost:5173...');
  await page.goto('http://localhost:5173', { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('canvas');
  console.log('Canvas loaded. Waiting 2.5s for initial 3D scene stabilization...');
  await sleep(2500);

  // Set up CDP Screencast
  const client = await page.target().createCDPSession();
  let frameIndex = 0;
  const frameMetadata = [];
  const startTime = Date.now();

  client.on('Page.screencastFrame', ({ data, sessionId }) => {
    const frameId = frameIndex++;
    const tRel = (Date.now() - startTime) / 1000;
    const filePath = path.join(FRAMES_DIR, `frame_${String(frameId).padStart(6, '0')}.jpg`);
    fs.writeFile(filePath, Buffer.from(data, 'base64'), () => {});
    frameMetadata.push({ index: frameId, time: tRel, file: filePath });
    client.send('Page.screencastFrameAck', { sessionId });
  });

  console.log('3. Starting screencast stream at 1600x900, quality 85...');
  await client.send('Page.startScreencast', { format: 'jpeg', quality: 85, everyNthFrame: 1 });

  console.log('>>> [00:00 - 00:05] Scene 1: Laboratory Hero Overview (5s)');
  await sleep(5000);

  console.log('>>> [00:05 - 00:18] Scene 2: Connected Production Line & Video Tour Flythrough (13s)');
  await page.keyboard.press('2');
  await sleep(1500);

  // Start Video Tour
  await clickButtonWithText(page, 'Video Tour');
  // Let the cinematic tour travel through the stations
  await sleep(11500);

  console.log('>>> [00:18 - 00:33] Flagship Interaction 1: Warwick NMC622 Calendering (15s)');
  // Exit tour and enter Calendering station
  await clickButtonWithText(page, 'Exit');
  await sleep(800);
  await clickButtonWithText(page, 'Enter flagship');
  await sleep(1800);

  // Begin AI decision
  await clickButtonWithText(page, 'choose the calendering conditions');
  await sleep(2200);

  // Run illustrated process response
  await clickButtonWithText(page, 'Run illustrated process response');
  // Wait for 3.2s GSAP choreography and result reveal
  await sleep(4000);

  // Bridge to Station 06
  await clickButtonWithText(page, 'Bridge to Measurement Station');
  await sleep(3500);

  console.log('>>> [00:33 - 00:44] Flagship Interaction 2: Drakopoulos Slot-Die Coating (11s)');
  // Switch to Drakopoulos Graphite Anode
  await page.keyboard.press('s');
  await sleep(1800);

  await clickButtonWithText(page, 'Enter flagship');
  await sleep(1500);

  await clickButtonWithText(page, 'select a recipe');
  await sleep(2000);

  await clickButtonWithText(page, 'Run illustrated process response');
  await sleep(4000);

  console.log('>>> [00:44 - 00:52] Scene 3: Inside the Electrode Microstructure (8s)');
  await page.keyboard.press('3');
  await sleep(2000);

  // Drag to rotate microstructure sample
  const mouse = page.mouse;
  await mouse.move(800, 450);
  await mouse.down();
  for (let step = 0; step < 25; step++) {
    await mouse.move(800 + step * 12, 450 - step * 3);
    await sleep(40);
  }
  await mouse.up();
  await sleep(1500);

  // Toggle auto morph
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const morphBtn = buttons.find(b => b.innerText && b.innerText.includes('Auto'));
    if (morphBtn) morphBtn.click();
  });
  await sleep(3500);

  console.log('>>> [00:52 - 00:57] Scene 4: AI Optimization Studio (5s)');
  await page.keyboard.press('4');
  await sleep(2000);

  // Click step buttons
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const s1 = buttons.find(b => b.innerText && b.innerText.trim() === 'Step 1');
    if (s1) s1.click();
  });
  await sleep(1200);

  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const s3 = buttons.find(b => b.innerText && b.innerText.trim() === 'Step 3');
    if (s3) s3.click();
  });
  await sleep(1800);

  console.log('>>> [00:57 - 01:00] Grand Finale: Clean Screen Panorama (3s)');
  await page.keyboard.press('1');
  await sleep(800);
  await page.keyboard.press('h');
  await sleep(2500);

  console.log('4. Stopping screencast and closing browser...');
  await client.send('Page.stopScreencast');
  await sleep(500);
  await browser.close();

  const metadataPath = path.join(FRAMES_DIR, 'metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(frameMetadata, null, 2));
  console.log(`Recorded total ${frameIndex} frames in ${(Date.now() - startTime) / 1000}s. Metadata saved to ${metadataPath}`);
})();
