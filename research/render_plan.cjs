const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const MarkdownIt = require('./preview_tools/node_modules/markdown-it');
const katex = require('./preview_tools/node_modules/katex');
const { chromium } = require('./preview_tools/node_modules/playwright-core');

const root = path.resolve(__dirname, '..');
const source = path.join(root, 'perturb_implementation_plan.md');
const destination = path.join(root, 'perturb_implementation_plan.html');
const qaDir = path.join(__dirname, 'qa_preview');
fs.mkdirSync(qaDir, { recursive: true });
let markdown = fs.readFileSync(source, 'utf8');
const sourceHash = crypto.createHash('sha256').update(markdown, 'utf8').digest('hex');
const equationTags = [...markdown.matchAll(/\\tag\{(\d+)\}/g)].map(match => Number(match[1]));
if (equationTags.some((tag, i) => tag !== i + 1)) throw new Error('Equation numbering is not consecutive');
const codeBlocks = [];
markdown = markdown.replace(/^```[^\n]*\n[\s\S]*?^```[ \t]*$/gm, (text) => {
  const id = codeBlocks.push(text) - 1;
  return `CODEFENCETOKEN${id}END`;
});
const expressions = [];
function renderMath(tex, displayMode) {
  const index = expressions.length;
  const html = katex.renderToString(tex.trim(), {
    displayMode, throwOnError: true, strict: 'error', trust: false,
    output: 'htmlAndMathml'
  });
  expressions.push({ tex: tex.trim(), displayMode, html });
  return displayMode ? `\n\nMATHBLOCKTOKEN${index}END\n\n` : `MATHINLINETOKEN${index}END`;
}
markdown = markdown.replace(/^\$\$[ \t]*\r?\n([\s\S]*?)^\$\$[ \t]*$/gm, (_, tex) => renderMath(tex, true));
markdown = markdown.replace(/(?<!\\)\$([^\n$]+?)(?<!\\)\$/g, (_, tex) => renderMath(tex, false));
codeBlocks.forEach((code, i) => { markdown = markdown.replace(`CODEFENCETOKEN${i}END`, code); });
let body = new MarkdownIt({ html: false, linkify: true, typographer: false }).render(markdown);
expressions.forEach((expression, i) => {
  if (expression.displayMode) {
    body = body.replace(`<p>MATHBLOCKTOKEN${i}END</p>`, `<div class="equation" data-equation="${i}">${expression.html}</div>`);
  } else {
    body = body.replaceAll(`MATHINLINETOKEN${i}END`, expression.html);
  }
});
if (/MATH(?:BLOCK|INLINE)TOKEN\d+END/.test(body)) throw new Error('Unreplaced math placeholder');
const dist = path.join(__dirname, 'preview_tools/node_modules/katex/dist');
let mathCss = fs.readFileSync(path.join(dist, 'katex.min.css'), 'utf8');
mathCss = mathCss.replace(/url\(([^)]+)\)/g, (_, raw) => {
  const filename = raw.replace(/["']/g, '');
  const font = path.join(dist, filename);
  const extension = path.extname(font).slice(1);
  return `url(data:font/${extension};base64,${fs.readFileSync(font).toString('base64')})`;
});
const css = `
*{box-sizing:border-box} body{margin:0;background:#edf1f3;color:#182830;font:16px/1.8 'Microsoft YaHei','Noto Sans CJK SC',sans-serif}
main{width:min(1100px,96vw);margin:28px auto;padding:40px 48px;background:white;border-top:5px solid #256276}
h1{font-size:29px;line-height:1.45;margin:0 0 20px}h2{font-size:23px;margin:34px 0 14px;padding-bottom:7px;border-bottom:1px solid #d7e2e6}h3{font-size:19px;margin:24px 0 10px}
p{margin:11px 0}a{color:#126282;text-decoration:underline}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:20px 0;table-layout:auto}
th,td{border:1px solid #cddadf;padding:8px 10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f4}
blockquote{border-left:4px solid #4b8596;background:#f0f6f8;margin:18px 0;padding:8px 18px}code{font:0.9em/1.65 Consolas,monospace;background:#eff3f5;padding:2px 4px;overflow-wrap:anywhere}
pre{background:#172a34;color:#e8f2f5;padding:18px;white-space:pre-wrap;border-radius:4px}pre code{background:none;padding:0;color:inherit}
.equation{padding:12px 18px;margin:18px 0;background:#f6f9fa;border:1px solid #e2e9ec;font-size:16px;overflow-x:auto}.katex-display{margin:8px 0}.katex{font-size:1.08em}
li{margin:5px 0}.preview-note{font-size:13px;color:#566973;margin:0 0 24px}.gallery main{width:1100px;margin:0 auto;padding:24px 36px}.gallery .equation{font-size:19px;margin:12px 0;padding:20px}.equation-label{font-size:14px;color:#355666}
@media print{body{background:white}main{width:auto;margin:0;padding:15px;border:0}h2,h3{break-after:avoid}.equation,table{break-inside:avoid}a{color:inherit}pre{color:black;background:#f5f5f5}}
`;
const wrap = (content, extra='') => `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Th2—Th17 项目实施方案</title><style>${mathCss}\n${css}</style></head><body class="${extra}"><main>${content}</main></body></html>`;
fs.writeFileSync(destination, wrap(`<p class="preview-note">离线公式预览 · 与 perturb_implementation_plan.md 同源生成 · 字体已内嵌，无需联网。</p>${body}`), 'utf8');

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1.5 });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(pathToFileURL(destination).href, { waitUntil: 'load' });
    await page.evaluate(() => document.fonts.ready);
    const layout = await page.evaluate(() => ({
      renderedMath: document.querySelectorAll('.katex').length,
      renderedDisplayMath: document.querySelectorAll('.equation').length,
      katexErrors: document.querySelectorAll('.katex-error').length,
      documentOverflow: document.documentElement.scrollWidth > innerWidth + 2,
      equationOverflow: [...document.querySelectorAll('.equation')].filter(e => e.scrollWidth > e.clientWidth + 2).map(e => e.dataset.equation),
      h2Count: document.querySelectorAll('h2').length,
      fontsStatus: document.fonts.status
    }));
    await page.screenshot({ path: path.join(qaDir, 'document_start.png') });
    const display = expressions.filter(e => e.displayMode);
    const shots = [];
    for (let start = 0; start < display.length; start += 4) {
      const chunk = display.slice(start, start + 4);
      const gallery = `<h1>公式预览 ${start+1}–${start+chunk.length}</h1>` + chunk.map((e,i) => `<p class="equation-label">行间公式 ${start+i+1}</p><div class="equation">${e.html}</div>`).join('');
      await page.setContent(wrap(gallery, 'gallery'), { waitUntil: 'load' });
      await page.evaluate(() => document.fonts.ready);
      const name = `formulas_${String(start+1).padStart(2,'0')}.png`;
      await page.screenshot({ path: path.join(qaDir,name), fullPage: true });
      shots.push(name);
    }
    const summary = {
      checked_at: new Date().toISOString(), source: path.basename(source), source_sha256: sourceHash,
      renderer: `KaTeX ${katex.version}`, browser: await browser.version(),
      expressions: expressions.length, displayExpressions: display.length, equationTags,
      layout, browserErrors: errors, screenshots: ['document_start.png', ...shots],
      passed: layout.katexErrors === 0 && !layout.documentOverflow && layout.equationOverflow.length === 0 && errors.length === 0 && layout.renderedMath === expressions.length
    };
    fs.writeFileSync(path.join(qaDir, 'validation.json'), JSON.stringify(summary,null,2));
    console.log(JSON.stringify(summary,null,2));
    if (!summary.passed) process.exitCode = 1;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
