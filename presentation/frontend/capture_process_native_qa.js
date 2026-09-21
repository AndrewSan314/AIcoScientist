import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const browserPath = process.env.SCREENSHOT_BROWSER || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const ffmpegPath = process.env.FFMPEG_PATH || 'C:\\Users\\ADMIN\\AppData\\Local\\CapCut\\Apps\\9.5.0.4050\\ffmpeg.exe';
const baseUrl = process.env.BASE_URL || 'http://127.0.0.1:5173';
const output = path.resolve('../screenshots/process-native-final');
fs.mkdirSync(output, { recursive: true });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function run() {
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: true, args: ['--no-sandbox', '--use-angle=swiftshader'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });

  const clickButton = async (text) => {
    await page.waitForFunction((needle) => [...document.querySelectorAll('button')].some((button) => button.offsetParent && button.textContent?.includes(needle)), { timeout: 10_000 }, text);
    const clicked = await page.evaluate((needle) => {
      const button = [...document.querySelectorAll('button')].find((item) => item.offsetParent && item.textContent?.includes(needle));
      button?.click();
      return Boolean(button);
    }, text);
    if (!clicked) throw new Error(`Missing button: ${text}`);
  };
  const shot = (name) => page.screenshot({ path: path.join(output, `${name}.png`) });
  const bodyText = () => page.evaluate(() => document.body.innerText);
  const assertText = async (text, present = true) => {
    const found = (await bodyText()).includes(text);
    if (found !== present) throw new Error(`${present ? 'Missing' : 'Prematurely revealed'} text: ${text}`);
  };

  await page.goto(baseUrl, { waitUntil: 'networkidle0', timeout: 30_000 });
  await sleep(1800);
  await clickButton('Connected Production Line');
  await sleep(1000);

  await clickButton('Warwick NMC622 Cathode');
  await clickButton('Calendering');
  await sleep(1400);
  const warwickRecorder = await page.screencast({ path: path.join(output, 'warwick-calendering-poc.webm'), ffmpegPath, fps: 24, scale: .75 });
  await shot('warwick-01-mechanism');
  await clickButton('Let AI choose the calendering conditions');
  await sleep(700);
  await assertText('0.6876', false);
  await shot('warwick-02-ai-decision');
  await clickButton('Run illustrated process response');
  await sleep(1100);
  await shot('warwick-03-process-mid-a');
  await sleep(1200);
  await shot('warwick-04-process-mid-b');
  await sleep(1500);
  await assertText('0.6876');
  await shot('warwick-05-result-reveal');
  await warwickRecorder.stop();

  await clickButton('Open advanced analysis');
  await sleep(1100);
  await assertText('EXP_07');
  await shot('warwick-06-studio-synchronized');
  await clickButton('Connected Production Line');
  await sleep(1000);
  await assertText('0.6876');
  await shot('warwick-07-return-preserved');

  await clickButton('Drakopoulos Graphite Anode');
  await sleep(800);
  await clickButton('Connected Production Line');
  await sleep(500);
  await clickButton('Coating');
  await sleep(1500);
  const drakoRecorder = await page.screencast({ path: path.join(output, 'drakopoulos-coating-poc.webm'), ffmpegPath, fps: 24, scale: .75 });
  await shot('drakopoulos-01-mechanism');
  await clickButton('Let AI select a recipe');
  await sleep(700);
  await assertText('263.89', false);
  await shot('drakopoulos-02-ai-decision');
  await clickButton('Run illustrated process response');
  await sleep(1300);
  await shot('drakopoulos-03-process-mid-a');
  await sleep(1400);
  await shot('drakopoulos-04-process-mid-b');
  await sleep(1700);
  await assertText('263.89');
  await shot('drakopoulos-05-result-reveal');
  await drakoRecorder.stop();

  await browser.close();
  if (errors.length) throw new Error(`Browser errors:\n${errors.join('\n')}`);
  console.log(`Process-native QA passed. Artifacts: ${output}`);
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
