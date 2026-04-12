#!/usr/bin/env node

// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import { readFile, writeFile } from 'fs/promises';
import { resolve } from 'path';
import { runResearch } from '../engine/orchestrator.js';
import { FREE } from '../engine/capabilities.js';

// ANSI color codes
const c = {
  reset:   '\x1b[0m',
  bold:    '\x1b[1m',
  dim:     '\x1b[2m',
  green:   '\x1b[32m',
  cyan:    '\x1b[36m',
  yellow:  '\x1b[33m',
  red:     '\x1b[31m',
  white:   '\x1b[37m',
  bgCyan:  '\x1b[46m',
};

const VERSION = '1.0.0';
const PRICING_URL = 'https://aurolabs.ai/#pricing';

// -----------------------------------------------
// Argument parsing
// -----------------------------------------------

const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h') || args.length === 0) {
  printHelp();
  process.exit(0);
}

if (args.includes('--version') || args.includes('-v')) {
  console.log(`aurum v${VERSION}`);
  process.exit(0);
}

// Parse flags
let question = '';
let language = null;
let depth = null;
let outputFile = null;
let iterations = null;
let template = null;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--language' || arg === '-l') {
    language = args[++i];
  } else if (arg === '--depth' || arg === '-d') {
    depth = args[++i];
  } else if (arg === '--output' || arg === '-o') {
    outputFile = args[++i];
  } else if (arg === '--iterations' || arg === '-i') {
    iterations = parseInt(args[++i]);
    if (isNaN(iterations) || iterations < 1) iterations = 2;
    if (iterations > 2) {
      console.log(`${c.yellow}Compact limited to 2 iterations.${c.reset}`);
      iterations = 2;
    }
  } else if (arg === '--template' || arg === '-t') {
    template = args[++i];
  } else if (arg === '--brief' || arg === '-b') {
    // Silently accepted — bottom line always generated
  } else if (!arg.startsWith('-')) {
    question = arg;
  }
}

// Reject deep mode
if (depth === 'deep') {
  console.error(`${c.red}Deep mode is not available in Compact. See https://aurolabs.ai for Aurum Deep.${c.reset}`);
  process.exit(1);
}

if (!question) {
  console.error(`${c.red}Error: No question provided.${c.reset}\n`);
  printHelp();
  process.exit(1);
}

// -----------------------------------------------
// Config loading
// -----------------------------------------------

async function loadConfig() {
  const config = {};

  // Try .aurum.yaml in cwd
  try {
    const yaml = await import('js-yaml');
    const yamlPath = resolve(process.cwd(), '.aurum.yaml');
    const content = await readFile(yamlPath, 'utf-8');
    const parsed = yaml.default.load(content);
    if (parsed) Object.assign(config, parsed);
  } catch {
    // No yaml file, continue
  }

  // Env vars override yaml
  if (process.env.OPENAI_API_KEY) {
    config.openai_api_key = process.env.OPENAI_API_KEY;
  }

  // CLI flags override everything
  if (language) config.language = language;
  if (depth) config.depth = depth;
  if (iterations) config.iterations = iterations;
  if (template) config.template = template;

  return config;
}

// -----------------------------------------------
// Terminal output helpers
// -----------------------------------------------

const spinnerFrames = ['|', '/', '-', '\\'];
let spinnerIdx = 0;
let spinnerInterval = null;
let currentLine = '';

function startSpinner(text) {
  currentLine = text;
  spinnerIdx = 0;
  process.stdout.write(`  ${c.cyan}${text}${c.reset} `);
  spinnerInterval = setInterval(() => {
    process.stdout.write(`\r  ${c.cyan}${currentLine}${c.reset} ${spinnerFrames[spinnerIdx++ % spinnerFrames.length]} `);
  }, 120);
}

function updateSpinner(text) {
  currentLine = text;
}

function stopSpinner(duration) {
  if (spinnerInterval) {
    clearInterval(spinnerInterval);
    spinnerInterval = null;
  }
  const timeStr = formatDuration(duration);
  process.stdout.write(`\r  ${currentLine}${' '.repeat(40)}\r  ${c.green}${currentLine}${c.reset}${c.dim}  ${timeStr}${c.reset}\n`);
}

function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}min`;
}

function progressBar(current, total, width = 20) {
  const filled = Math.round((current / total) * width);
  const empty = width - filled;
  return '[' + '|'.repeat(filled) + ' '.repeat(empty) + ']';
}

// -----------------------------------------------
// Main
// -----------------------------------------------

async function main() {
  const config = await loadConfig();

  if (!config.openai_api_key) {
    console.error(`
${c.red}${c.bold}No OpenAI API key found.${c.reset}

Set it via environment variable:
  ${c.cyan}export OPENAI_API_KEY=sk-...${c.reset}

Or create ${c.cyan}.aurum.yaml${c.reset} in your project:
  ${c.dim}openai_api_key: sk-...${c.reset}

Get a key at ${c.cyan}https://platform.openai.com${c.reset}
`);
    process.exit(1);
  }

  const output = outputFile || 'rapport.md';
  const maxIter = Math.min(config.iterations || 2, FREE.maxIterations);

  console.log(`
${c.bold}aurum${c.reset} ${c.dim}v${VERSION}${c.reset}

  ${c.bold}Question:${c.reset} ${question}
  ${c.dim}Language: ${config.language || 'en'} | Depth: ${config.depth || 'standard'} | Iterations: ${maxIter} | Output: ${output}${c.reset}
  ${c.dim}Template: ${config.template || 'auto'}${c.reset}
`);

  const phaseTimers = {};

  function onProgress(phase, data) {
    if (data.status === 'start') {
      phaseTimers[phase] = Date.now();

      switch (phase) {
        case 'plan':
          startSpinner('Planning research...');
          break;
        case 'search':
          startSpinner(`Searching sources... ${progressBar(0, data.total)} 0/${data.total}`);
          break;
        case 'reduce':
          startSpinner(`Analyzing data... ${progressBar(0, data.sections)} 0/${data.sections} sections`);
          break;
        case 'write':
          startSpinner('Writing report...');
          break;
        case 'verify':
          startSpinner('Verifying sources...');
          break;
        case 'director':
          startSpinner(`Director review (pass ${data.iteration})...`);
          break;
        case 'improve':
          startSpinner(`Improving report (pass ${data.iteration})...`);
          break;
        case 'brief':
          startSpinner('Generating bottom line...');
          break;
      }
    } else if (data.status === 'progress') {
      switch (phase) {
        case 'search':
          updateSpinner(`Searching sources... ${progressBar(data.completed, data.total)} ${data.completed}/${data.total}`);
          break;
      }
    } else if (data.status === 'done') {
      const elapsed = Date.now() - (phaseTimers[phase] || Date.now());
      switch (phase) {
        case 'template':
          // Silent -- just log template choice
          break;
        case 'plan':
          stopSpinner(elapsed);
          break;
        case 'search':
          currentLine = `Searching sources... ${progressBar(data.completed, data.completed)} ${data.completed} sources`;
          stopSpinner(elapsed);
          break;
        case 'reduce':
          currentLine = `Analyzing data... ${progressBar(data.sections, data.sections)} ${data.sections} sections`;
          stopSpinner(elapsed);
          break;
        case 'write':
          currentLine = `Writing report... ${data.sections} sections`;
          stopSpinner(elapsed);
          break;
        case 'verify':
          stopSpinner(elapsed);
          break;
        case 'director':
          currentLine = `Director review: ${data.score}/100 (${data.action})`;
          stopSpinner(elapsed);
          break;
        case 'improve':
          stopSpinner(elapsed);
          break;
        case 'brief':
          currentLine = 'Bottom line generated';
          stopSpinner(elapsed);
          break;
      }
    }
  }

  const startTime = Date.now();

  try {
    const result = await runResearch(question, { openai: config.openai_api_key }, {
      language: config.language || 'en',
      depth: config.depth || 'standard',
      max_iterations: maxIter,
      template: config.template || 'auto',
      capabilities: FREE,
      onProgress,
    });

    // Save report
    const outputPath = resolve(process.cwd(), output);
    await writeFile(outputPath, result.markdown, 'utf-8');

    const totalTime = Date.now() - startTime;
    const costEstimate = result.costs.total_cost_usd;

    // Bottom line display
    const bl = result.brief;
    const bottomLineStr = bl && bl.bottom_line
      ? `${bl.bottom_line} [${bl.confidence?.kent_term || '?'}, ${bl.confidence?.percentage || '?'}%]`
      : '';

    console.log(`
  ${c.green}${c.bold}Report saved:${c.reset} ${c.cyan}./${output}${c.reset}

  ${bottomLineStr ? `${c.bold}Bottom line:${c.reset} ${bottomLineStr}\n` : ''}
  ${c.dim}Words:      ${result.metadata.word_count.toLocaleString()}${c.reset}
  ${c.dim}Sources:    ${result.metadata.source_count}${c.reset}
  ${c.dim}Template:   ${result.metadata.template_used}${c.reset}
  ${c.dim}Iterations: ${result.metadata.iterations}${c.reset}
  ${c.dim}Director:   ${result.metadata.director_score}/100${c.reset}
  ${c.dim}Time:       ${formatDuration(totalTime)}${c.reset}
  ${c.dim}Cost:       ~$${costEstimate.toFixed(2)}${c.reset}
`);

  } catch (err) {
    if (spinnerInterval) {
      clearInterval(spinnerInterval);
      spinnerInterval = null;
    }
    console.error(`\n  ${c.red}Error: ${err.message}${c.reset}\n`);
    if (process.env.DEBUG) {
      console.error(err.stack);
    }
    process.exit(1);
  }
}

function printHelp() {
  console.log(`
${c.bold}aurum${c.reset} ${c.dim}v${VERSION}${c.reset}
The difference between guessing and knowing.
Agentic research engine. Searches reactively. Thinks before it writes.

${c.bold}USAGE${c.reset}
  aurum "question"                        Run research, save report
  aurum "question" --depth quick          Faster, shorter report
  aurum "question" --template NAME        Force template

${c.bold}OPTIONS${c.reset}
  -l, --language <code>      Output language (en, sv, de, fr, ...)
  -d, --depth <level>        quick | standard
  -o, --output <file>        Output filename (default: rapport.md)
  -i, --iterations <n>       Director loop iterations (default: 2, max: 2)
  -t, --template <name>      Force template (default: auto-detect)
  -h, --help                 Show this help
  -v, --version              Show version

${c.bold}TEMPLATES${c.reset} (auto-selected, or use --template)
  scenario-analysis    Future scenarios, geopolitical, policy impact
  market-research      Market sizing, segmentation, TAM/SAM/SOM
  competitor-audit     Competitive landscape, positioning

${c.bold}DEPTH MODES${c.reset}
  quick       5-10 min    2,000-4,000 words    10-15 searches   ~$3
  standard    15-30 min   5,000-15,000 words   20-30 searches   ~$5-8

  ${c.dim}Deep mode requires Pro. See ${c.cyan}${PRICING_URL}${c.reset}

${c.bold}SETUP${c.reset}
  export OPENAI_API_KEY=sk-...
  ${c.dim}Or create .aurum.yaml with openai_api_key${c.reset}

${c.bold}EXAMPLES${c.reset}
  aurum "What happens if the EU bans diesel cars by 2030?"
  aurum "AI chip export controls impact" --language sv
  aurum "Bitcoin ETF market analysis" --depth quick -o btc.md
  aurum "Nordic fintech competitive landscape" --template competitor-audit

${c.bold}MODELS${c.reset}
  gpt-5.4-mini (research) + gpt-5.4 (writing/director)

${c.bold}DEEP${c.reset}
  Need deeper analysis? Aurum Deep: EUR 499/report.
  ${c.cyan}https://aurolabs.ai${c.reset}

${c.bold}MORE${c.reset}
  Source:  ${c.cyan}https://github.com/aurolabs-ai/aurum${c.reset}
  Home:   ${c.cyan}https://aurolabs.ai${c.reset}
`);
}

main();
