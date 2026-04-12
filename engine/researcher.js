// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';
import fs from 'fs';
import { loadTemplate } from './template-matcher.js';
import { reduceData } from './reduce.js';

const RESEARCH_MODEL = 'gpt-4.1-mini';
const WRITING_MODEL = 'gpt-4.1';
const PLANNING_MODEL = 'gpt-4.1-mini';
const VERIFICATION_MODEL = 'gpt-4.1-mini';
const ASSESSMENT_MODEL = 'gpt-4.1-mini';

const PRICING = {
  'gpt-4.1':      { in: 2.0,  out: 8.0  },
  'gpt-4.1-mini': { in: 0.4,  out: 1.6  },
};

const MAX_SEARCH_TURNS = 50;
const MAX_WRITE_TOKENS = 16000;
const MAX_RESEARCH_ROUNDS = 3;
const SEARCH_BATCH_SIZE = 5;  // Reduced from 10 to avoid overwhelming single thread
const MAX_SCRATCHPAD_CHARS = 100_000;  // Truncate beyond this — diminishing returns
const RSS_LIMIT_MB = 200;

// OpenAI web_search_preview pricing for non-reasoning models (gpt-4.1, gpt-4.1-mini):
// $25 per 1,000 calls = $0.025 per call. Search content tokens are free.
const WEB_SEARCH_COST_PER_CALL = 0.025;

/** Race an API call against a 30s timeout. Returns the response or throws. */
async function withTimeout(promise, label = 'API call', timeoutMs = 30000) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs)
    ),
  ]);
}

/** Check RSS memory. If over limit, try GC and return a reduced batch size. */
function checkMemoryPressure(normalBatchSize) {
  const rssMb = process.memoryUsage.rss() / (1024 * 1024);
  if (rssMb > RSS_LIMIT_MB) {
    console.log(`[Researcher-OpenAI] Memory pressure: RSS=${rssMb.toFixed(0)}MB > ${RSS_LIMIT_MB}MB limit`);
    if (global.gc) {
      try { global.gc(); } catch {}
      console.log(`[Researcher-OpenAI] Forced GC, RSS now ${(process.memoryUsage.rss() / (1024 * 1024)).toFixed(0)}MB`);
    }
    return Math.max(2, Math.floor(normalBatchSize / 2));
  }
  return normalBatchSize;
}

/** Generate a checkpoint path for a given job. */
function checkpointPath(jobId, phase) {
  return `/tmp/aurum-${jobId}-${phase}`;
}

/** Try to load a checkpoint. Returns null if not found or invalid. */
function loadCheckpoint(jobId, phase) {
  const p = checkpointPath(jobId, phase);
  try {
    if (fs.existsSync(p)) {
      const data = fs.readFileSync(p, 'utf-8');
      console.log(`[Researcher-OpenAI] Loaded checkpoint: ${p} (${data.length} chars)`);
      if (phase === 'evidence') return JSON.parse(data);
      return data;
    }
  } catch (err) {
    console.log(`[Researcher-OpenAI] Checkpoint load failed (${p}): ${err.message}`);
  }
  return null;
}

/** Save a checkpoint. */
function saveCheckpoint(jobId, phase, data) {
  const p = checkpointPath(jobId, phase);
  try {
    const content = typeof data === 'string' ? data : JSON.stringify(data);
    fs.writeFileSync(p, content);
    console.log(`[Researcher-OpenAI] Saved checkpoint: ${p} (${content.length} chars)`);
  } catch (err) {
    console.log(`[Researcher-OpenAI] Checkpoint save failed (${p}): ${err.message}`);
  }
}

/** Clean up temp checkpoint files for a job. */
function cleanCheckpoints(jobId) {
  for (const phase of ['scratchpad', 'evidence']) {
    const p = checkpointPath(jobId, phase);
    try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {}
  }
}

/**
 * Multi-phase OpenAI research agent with iterative assessment.
 *
 * Phase 1: Research Planning (mini) - generates search queries + section plan
 * Phase 2: Iterative Data Collection with Assessment
 *   - Round 1: Execute planned queries, assess quality, identify gaps
 *   - Round 2+: Generate targeted queries for weak/missing sections, re-search
 * Phase 3: Reduce (deduplicate + structure evidence by section)
 * Phase 4: Sectional Report Writing with paragraph-level self-review
 * Phase 5: Source Verification
 */
export async function researchOpenAI(question, openaiKey, options = {}) {
  const {
    templateName,
    language = 'en',
    depth = 'standard',
    timeLimitMs = 55 * 60 * 1000,
    costTracker = {},
    onProgress = () => {},
  } = options;

  const startTime = Date.now();
  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCost = 0;
  let totalTurns = 0;
  let totalWebSearchCalls = 0;

  const client = new OpenAI({ apiKey: openaiKey });

  const templateContent = await loadTemplate(templateName);
  const filledTemplate = fillTemplate(templateContent, question, language);
  const langName = language === 'en' ? 'English' : language === 'sv' ? 'Swedish' : language;

  const depthConfig = getDepthConfig(depth);

  // --- Research context: accumulated across all phases ---
  const ctx = {
    question,
    scratchpad: `# Research Scratchpad\n## Question: ${question}\n## Language: ${langName}\n\n`,
    assessments: [],
    gaps: [],
    rejectedSources: [],   // domains/URLs to avoid
    sectionCoverage: {},   // per-section: 'strong' | 'weak' | 'missing'
    searchCount: 0,
    allQueriesRun: new Set(),
  };

  function elapsed() { return Date.now() - startTime; }
  function timeLeft() { return timeLimitMs - elapsed(); }
  function hasTime(reserveMs = 120_000) { return timeLeft() > reserveMs; }

  // HARD DEADLINE: absolute max time. If exceeded, skip to write with whatever we have.
  const HARD_DEADLINE_MS = timeLimitMs + 120_000; // time_limit + 2min grace
  function pastDeadline() {
    if (elapsed() > HARD_DEADLINE_MS) {
      console.log(`[Researcher-OpenAI] HARD DEADLINE EXCEEDED (${Math.round(elapsed()/1000)}s > ${Math.round(HARD_DEADLINE_MS/1000)}s). Forcing write phase.`);
      return true;
    }
    return false;
  }

  function trackUsage(model, usage) {
    if (!usage) return;
    const tokIn = usage.input_tokens || usage.prompt_tokens || 0;
    const tokOut = usage.output_tokens || usage.completion_tokens || 0;
    const p = PRICING[model] || PRICING['gpt-4.1'];
    const cost = (tokIn * p.in / 1_000_000) + (tokOut * p.out / 1_000_000);
    totalTokensIn += tokIn;
    totalTokensOut += tokOut;
    totalCost += cost;
    totalTurns++;
    return { tokIn, tokOut, cost };
  }

  // Generate a stable job ID for checkpointing (hash of question)
  const jobId = question.replace(/[^a-zA-Z0-9]/g, '').slice(0, 40) || 'default';

  // ========== PHASE 1: Research Planning ==========
  onProgress('plan', { status: 'start' });
  console.log(`[Researcher-OpenAI] === PHASE 1: Research Planning ===`);
  console.log(`[Researcher-OpenAI] Question: "${question.slice(0, 100)}..."`);
  console.log(`[Researcher-OpenAI] Template: ${templateName}, Depth: ${depth}, Language: ${language}`);

  let plan;
  try {
    plan = await withTimeout(planResearch(client, question, filledTemplate, langName, depthConfig), 'planResearch', 30000);
  } catch (err) {
    console.log(`[Researcher-OpenAI] Planning failed: ${err.message}, using fallback plan`);
    plan = buildFallbackPlan(question, depthConfig);
  }
  console.log(`[Researcher-OpenAI] Plan: ${plan.sections.length} sections, ${plan.search_queries.length} queries, ${plan.key_questions.length} sub-questions`);
  onProgress('plan', { status: 'done' });

  // Initialize section coverage as 'missing'
  for (const s of plan.sections) {
    ctx.sectionCoverage[s] = 'missing';
  }

  // ========== PHASE 2: Iterative Data Collection ==========
  console.log(`[Researcher-OpenAI] === PHASE 2: Iterative Data Collection ===`);
  const totalPlannedSearches = plan.search_queries.length + plan.key_questions.length;
  onProgress('search', { status: 'start', total: totalPlannedSearches });

  // Check for scratchpad checkpoint (resume from crash)
  const cachedScratchpad = loadCheckpoint(jobId, 'scratchpad');
  if (cachedScratchpad) {
    console.log(`[Researcher-OpenAI] Resuming from scratchpad checkpoint (${cachedScratchpad.length} chars)`);
    ctx.scratchpad = cachedScratchpad;
    ctx.searchCount = (cachedScratchpad.match(/### Search:/g) || []).length;
  } else {
    // Reserve time: 6min for writing, 1min for verification
    // Reserve time for write+verify+brief — proportional to time limit, min 2min, max 7min
  const writeReserveMs = Math.max(120_000, Math.min(420_000, timeLimitMs * 0.3));

    // --- Round 1: Execute planned queries ---
    const round1Queries = plan.search_queries.slice(0, Math.min(plan.search_queries.length, MAX_SEARCH_TURNS));
    await executeSearchRound(client, round1Queries, ctx, SEARCH_BATCH_SIZE, writeReserveMs, hasTime, trackUsage, 1);

    // Also search key_questions in parallel if we have time
    if (hasTime(writeReserveMs) && plan.key_questions.length > 0) {
      console.log(`[Researcher-OpenAI] --- Searching key sub-questions ---`);
      await executeSearchRound(client, plan.key_questions, ctx, SEARCH_BATCH_SIZE, writeReserveMs, hasTime, trackUsage, 1, 'Sub-Q');
    }

    // Save checkpoint after round 1
    saveCheckpoint(jobId, 'scratchpad', ctx.scratchpad);

    // --- Post-round assessment: quality filter + coverage check ---
    for (let round = 2; round <= MAX_RESEARCH_ROUNDS; round++) {
      if (pastDeadline() || !hasTime(writeReserveMs + 60_000)) {
        console.log(`[Researcher-OpenAI] No time for additional research rounds`);
        break;
      }

      console.log(`[Researcher-OpenAI] === Assessment after round ${round - 1} ===`);
      let assessment;
      try {
        assessment = await withTimeout(assessCoverage(client, ctx, plan.sections, trackUsage), 'assessCoverage', 30000);
      } catch (err) {
        console.log(`[Researcher-OpenAI] Assessment failed: ${err.message}, skipping further rounds`);
        break;
      }
      ctx.assessments.push(assessment);

      if (assessment.rejectedDomains?.length > 0) {
        ctx.rejectedSources.push(...assessment.rejectedDomains);
        console.log(`[Researcher-OpenAI] Rejected domains: ${assessment.rejectedDomains.join(', ')}`);
      }

      // Update section coverage
      for (const s of plan.sections) {
        if (assessment.sectionCoverage?.[s]) {
          ctx.sectionCoverage[s] = assessment.sectionCoverage[s];
        }
      }

      const weakSections = Object.entries(ctx.sectionCoverage).filter(([, v]) => v === 'weak' || v === 'missing');
      console.log(`[Researcher-OpenAI] Coverage: ${Object.values(ctx.sectionCoverage).filter(v => v === 'strong').length} strong, ${weakSections.length} weak/missing`);

      // If coverage is good enough, stop
      if (weakSections.length === 0) {
        console.log(`[Researcher-OpenAI] All sections have strong coverage, skipping round ${round}`);
        break;
      }

      // Generate targeted queries for weak sections — cap based on time remaining
      const additionalQueries = assessment.additionalQueries || [];
      const filteredQueries = additionalQueries.filter(q => !ctx.allQueriesRun.has(q));
      if (filteredQueries.length === 0) {
        console.log(`[Researcher-OpenAI] No new queries to run, stopping research rounds`);
        break;
      }
      // Cap queries based on time: ~30s per query in batch of 5 = ~6s effective per query
      const timeForQueries = Math.max(0, timeLeft() - writeReserveMs - 60_000);
      const maxQueries = Math.max(5, Math.floor(timeForQueries / 6000));
      const newQueries = filteredQueries.slice(0, maxQueries);
      if (newQueries.length < filteredQueries.length) {
        console.log(`[Researcher-OpenAI] Capped queries from ${filteredQueries.length} to ${newQueries.length} (${Math.round(timeForQueries/1000)}s available)`);
      }

      console.log(`[Researcher-OpenAI] === Research Round ${round}: ${newQueries.length} targeted queries ===`);
      await executeSearchRound(client, newQueries, ctx, SEARCH_BATCH_SIZE, writeReserveMs, hasTime, trackUsage, round);

      // Save checkpoint after each round
      saveCheckpoint(jobId, 'scratchpad', ctx.scratchpad);
    }
  }

  // Truncate scratchpad if too large (diminishing returns beyond 100K chars)
  if (ctx.scratchpad.length > MAX_SCRATCHPAD_CHARS) {
    console.log(`[Researcher-OpenAI] Truncating scratchpad from ${ctx.scratchpad.length} to ${MAX_SCRATCHPAD_CHARS} chars`);
    ctx.scratchpad = ctx.scratchpad.slice(0, 5000) + '\n...[truncated middle]...\n' + ctx.scratchpad.slice(-(MAX_SCRATCHPAD_CHARS - 5000));
  }

  const scratchpadWords = ctx.scratchpad.split(/\s+/).length;
  console.log(`[Researcher-OpenAI] Data collection complete: ${ctx.searchCount} successful searches, ${scratchpadWords} words in scratchpad`);
  onProgress('search', { status: 'done', completed: ctx.searchCount, total: ctx.searchCount });

  // ========== PHASE 3: REDUCE ==========
  onProgress('reduce', { status: 'start', sections: plan.sections.length });
  console.log(`[Researcher-OpenAI] === PHASE 3: Reduce (${plan.sections.length} sections) ===`);
  const reduceStartTime = Date.now();
  const reduceCosts = {};

  // Check for evidence checkpoint (resume from crash)
  let evidencePack = loadCheckpoint(jobId, 'evidence');
  if (evidencePack) {
    console.log(`[Researcher-OpenAI] Resuming from evidence checkpoint (${Object.keys(evidencePack).length} sections)`);
  } else {
    try {
      evidencePack = await reduceData(ctx.scratchpad, plan.sections, client.apiKey, reduceCosts);
      if (reduceCosts.tokens_in) totalTokensIn += reduceCosts.tokens_in;
      if (reduceCosts.tokens_out) totalTokensOut += reduceCosts.tokens_out;
      if (reduceCosts.cost_usd) totalCost += reduceCosts.cost_usd;
      // Save evidence checkpoint
      saveCheckpoint(jobId, 'evidence', evidencePack);
    } catch (err) {
      console.log(`[Researcher-OpenAI] Reduce failed: ${err.message}, using empty evidence packs`);
      evidencePack = {};
      for (const s of plan.sections) {
        evidencePack[s] = { claims: [], key_stats: [], contradictions: [], sources: [] };
      }
    }
  }
  console.log(`[Researcher-OpenAI] Reduce complete: ${(Date.now() - reduceStartTime) / 1000}s, $${(reduceCosts.cost_usd || 0).toFixed(4)}`);
  onProgress('reduce', { status: 'done', sections: plan.sections.length });

  // ========== PHASE 4: Sectional Report Writing with Self-Review ==========
  onProgress('write', { status: 'start' });
  console.log(`[Researcher-OpenAI] === PHASE 4: Sectional Report Writing (${plan.sections.length} sections) ===`);
  const writeStartTime = Date.now();

  let fullReport = '';
  let writtenSections = [];

  // Write intro/methodology first
  {
    console.log(`[Researcher-OpenAI] Writing intro + methodology...`);
    try {
      const introText = await withTimeout(writeSectionDraft(client, {
        sectionName: 'Introduction and Methodology',
        evidence: null,
        question,
        template: filledTemplate,
        langName,
        depthConfig,
        allSections: plan.sections,
        previousSections: '',
        isIntro: true,
        ctx,
      }), 'writeIntro', 60000);
      if (introText.text) {
        fullReport += introText.text + '\n\n';
        trackUsage(WRITING_MODEL, introText.usage);
        writtenSections.push('Introduction');
        console.log(`[Researcher-OpenAI]   Intro: ${introText.text.split(/\s+/).length} words`);
      }
    } catch (err) {
      console.log(`[Researcher-OpenAI]   Intro failed: ${err.message}, continuing without intro`);
    }
  }

  // Write sections: first 3 sequential (build narrative), rest parallel, conclusion last
  const sectionsWithEvidence = plan.sections.filter(name => {
    const ev = evidencePack[name];
    return ev && (ev.claims.length > 0 || ev.key_stats.length > 0);
  });

  const SEQUENTIAL_COUNT = Math.min(3, sectionsWithEvidence.length);
  const sequentialSections = sectionsWithEvidence.slice(0, SEQUENTIAL_COUNT);
  const parallelSections = sectionsWithEvidence.slice(SEQUENTIAL_COUNT);

  // Phase 4a: Sequential sections (build foundation narrative)
  console.log(`[Researcher-OpenAI] Writing ${sequentialSections.length} foundation sections sequentially...`);
  for (const sectionName of sequentialSections) {
    if (!hasTime(90_000)) break;
    const evidence = evidencePack[sectionName];
    console.log(`[Researcher-OpenAI]   Writing "${sectionName}" (${evidence.claims.length} claims, ${evidence.key_stats.length} stats)...`);
    try {
      let draft = await withTimeout(writeSectionDraft(client, {
        sectionName, evidence, question, template: filledTemplate, langName, depthConfig,
        allSections: plan.sections, previousSections: fullReport.slice(-3000), isIntro: false, ctx,
      }), `writeSection:${sectionName}`, 60000);
      if (draft.text && draft.text.trim().length > 100) {
        trackUsage(WRITING_MODEL, draft.usage);
        // --- Paragraph-level self-review ---
        if (hasTime(90_000)) {
          try {
            draft.text = await withTimeout(
              reviewAndFixSection(client, draft.text, fullReport, sectionName, ctx, trackUsage),
              `reviewSection:${sectionName}`, 30000
            );
          } catch (reviewErr) {
            console.log(`[Researcher-OpenAI]   Review of "${sectionName}" failed: ${reviewErr.message}, keeping draft`);
          }
        }
        fullReport += draft.text + '\n\n';
        writtenSections.push(sectionName);
        console.log(`[Researcher-OpenAI]   "${sectionName}": ${draft.text.split(/\s+/).length} words (reviewed)`);
      }
    } catch (err) {
      console.log(`[Researcher-OpenAI]   "${sectionName}": FAILED - ${err.message}, skipping`);
    }
  }

  // Phase 4b: Parallel sections (independent analysis, share foundation context)
  if (parallelSections.length > 0 && hasTime(90_000)) {
    const WRITE_BATCH = 4;
    const foundationContext = fullReport.slice(-4000);
    console.log(`[Researcher-OpenAI] Writing ${parallelSections.length} analysis sections in parallel (batches of ${WRITE_BATCH})...`);

    for (let bStart = 0; bStart < parallelSections.length; bStart += WRITE_BATCH) {
      if (!hasTime(90_000)) {
        console.log(`[Researcher-OpenAI] Time limit, skipping remaining sections`);
        break;
      }
      const batch = parallelSections.slice(bStart, bStart + WRITE_BATCH);

      const results = await Promise.allSettled(batch.map(async (sectionName) => {
        const evidence = evidencePack[sectionName];
        const draft = await writeSectionDraft(client, {
          sectionName, evidence, question, template: filledTemplate, langName, depthConfig,
          allSections: plan.sections, previousSections: foundationContext, isIntro: false, ctx,
        });
        if (draft.text && draft.text.trim().length > 100) {
          trackUsage(WRITING_MODEL, draft.usage);
          // --- Paragraph-level self-review (parallel with other sections) ---
          if (hasTime(60_000)) {
            draft.text = await reviewAndFixSection(client, draft.text, fullReport, sectionName, ctx, trackUsage);
          }
          return { sectionName, text: draft.text };
        }
        return null;
      }));

      for (const result of results) {
        if (result.status !== 'fulfilled' || !result.value) continue;
        const { sectionName, text } = result.value;
        fullReport += text + '\n\n';
        writtenSections.push(sectionName);
        console.log(`[Researcher-OpenAI]   "${sectionName}": ${text.split(/\s+/).length} words (reviewed)`);
      }
    }
  }

  // Write conclusion + source list
  if (hasTime(60_000)) {
    console.log(`[Researcher-OpenAI] Writing conclusion + source list...`);
    try {
      const conclusion = await writeSectionDraft(client, {
        sectionName: 'Conclusion and Sources',
        evidence: null,
        question,
        template: filledTemplate,
        langName,
        depthConfig,
        allSections: plan.sections,
        previousSections: fullReport.slice(-5000),
        isIntro: false,
        isConclusion: true,
        fullReportSoFar: fullReport,
        ctx,
      });
      if (conclusion.text) {
        fullReport += conclusion.text;
        trackUsage(WRITING_MODEL, conclusion.usage);
        console.log(`[Researcher-OpenAI]   Conclusion: ${conclusion.text.split(/\s+/).length} words`);
      }
    } catch (err) {
      console.log(`[Researcher-OpenAI]   Conclusion failed: ${err.message}`);
    }
  }

  const reportWords = fullReport.split(/\s+/).length;
  console.log(`[Researcher-OpenAI] Report written: ${reportWords} words across ${writtenSections.length} sections in ${((Date.now() - writeStartTime) / 1000).toFixed(1)}s`);
  onProgress('write', { status: 'done', sections: writtenSections.length });

  // ========== PHASE 5: Source Verification ==========
  if (hasTime(60_000)) {
    onProgress('verify', { status: 'start' });
    console.log(`[Researcher-OpenAI] === PHASE 5: Source Verification ===`);
    try {
      const verified = await verifyReport(client, fullReport, ctx.scratchpad);
      if (verified.text && verified.text.trim().length > fullReport.length * 0.8) {
        fullReport = verified.text;
        trackUsage(VERIFICATION_MODEL, verified.usage);
        console.log(`[Researcher-OpenAI] Verification complete, report updated`);
      } else {
        console.log(`[Researcher-OpenAI] Verification returned insufficient output, keeping original`);
        if (verified.usage) trackUsage(VERIFICATION_MODEL, verified.usage);
      }
    } catch (err) {
      console.log(`[Researcher-OpenAI] Verification failed: ${err.message}, keeping original report`);
    }
    onProgress('verify', { status: 'done' });
  } else {
    console.log(`[Researcher-OpenAI] Skipping verification (time limit)`);
  }

  // ========== Finalize: fix fabricated source counts ==========
  fullReport = cleanReport(fullReport);

  // Count ACTUAL unique URLs in the report
  const actualUrls = fullReport.match(/\]\(https?:\/\/[^)]+\)/g) || [];
  const actualSourceCount = [...new Set(actualUrls.map(m => m.slice(2, -1).split('#')[0].split('?')[0]))].length;

  // Replace any fabricated source count claims with actual count
  fullReport = fullReport.replace(
    /(?:over|more than|approximately|nearly|draws? (?:on|from|exclusively from))\s+(\d{2,4})\s+(?:unique\s+)?(?:sources|references|citations)/gi,
    (match, num) => {
      const claimed = parseInt(num);
      if (claimed > actualSourceCount * 1.2) {
        console.log(`[Researcher-OpenAI] Fixed fabricated source count: ${claimed} -> ${actualSourceCount}`);
        return match.replace(num, String(actualSourceCount));
      }
      return match;
    }
  );
  // Also fix standalone "N sources" patterns in methodology sections
  fullReport = fullReport.replace(
    /(\d{2,4})\s+(?:unique,?\s+)?(?:recent,?\s+)?(?:and\s+)?(?:authoritative\s+)?sources/gi,
    (match, num) => {
      const claimed = parseInt(num);
      if (claimed > actualSourceCount * 1.2) {
        console.log(`[Researcher-OpenAI] Fixed fabricated source claim: "${match}" -> ${actualSourceCount} sources`);
        return match.replace(num, String(actualSourceCount));
      }
      return match;
    }
  );

  const finalWords = fullReport.split(/\s+/).length;
  const elapsedSec = elapsed() / 1000;
  console.log(`[Researcher-OpenAI] === COMPLETE ===`);
  const webSearchCostTotal = totalWebSearchCalls * WEB_SEARCH_COST_PER_CALL;
  console.log(`[Researcher-OpenAI] ${finalWords} words | ${ctx.searchCount} searches (${totalWebSearchCalls} web calls = $${webSearchCostTotal.toFixed(2)}) | ${totalTurns} API turns | ${elapsedSec.toFixed(0)}s | $${totalCost.toFixed(2)}`);
  console.log(`[Researcher-OpenAI] Research rounds: ${ctx.assessments.length + 1} | Rejected domains: ${ctx.rejectedSources.length}`);

  // Update cost tracker for orchestrator
  costTracker.model = RESEARCH_MODEL;
  costTracker.tokens_in = totalTokensIn;
  costTracker.tokens_out = totalTokensOut;
  costTracker.cost_usd = totalCost;
  costTracker.turn_count = totalTurns;
  costTracker.search_count = ctx.searchCount;
  costTracker.web_search_calls = totalWebSearchCalls;
  costTracker.web_search_cost_usd = webSearchCostTotal;

  if (fullReport.trim().length < 500) {
    throw new Error('Research produced insufficient output (less than 500 characters)');
  }

  // Clean up checkpoints on success
  cleanCheckpoints(jobId);

  return fullReport.trim();


  // ====================================================================
  // Internal phase functions (closures over client, trackUsage, etc.)
  // ====================================================================

  async function planResearch(client, question, template, langName, depthCfg) {
    const queryCount = depthCfg.queryCount;

    const response = await withTimeout(client.chat.completions.create({
      model: PLANNING_MODEL,
      temperature: 0.3,
      max_completion_tokens: 4000,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: `You are a research planning assistant. Given a research question and a report template, produce a JSON research plan.

The plan must include:
1. "sections" - array of section titles for the final report, based on the template structure
2. "search_queries" - array of ${queryCount[0]}-${queryCount[1]} specific web search queries. CRITICAL RULES for query quality:

   AT LEAST 30-40% of queries MUST target authoritative sources using site: prefix:
   - Academic: site:ncbi.nlm.nih.gov, site:pubmed.gov, site:scholar.google.com, site:arxiv.org
   - Government: site:gov, site:europa.eu, site:who.int, site:fda.gov, site:oecd.org
   - Institutional: site:imf.org, site:worldbank.org, site:un.org
   - Industry reports: site:mckinsey.com, site:bcg.com, site:gartner.com, site:statista.com
   - Financial: site:bloomberg.com, site:reuters.com, site:ft.com
   - Data: site:data.worldbank.org, site:fred.stlouisfed.org, site:eurostat.ec.europa.eu

   The remaining queries should be:
   - Broad overview queries (no site: prefix)
   - Specific data queries with year markers (2024, 2025, 2026)
   - Contrarian/counter-argument queries
   - Recent news queries

   NEVER rely primarily on SEO-optimized blogs, content farms, or low-authority domains.

3. "key_questions" - array of 5-10 sub-questions that need answering

Make queries specific and searchable. Include year markers. Vary query formulations.
Output ONLY valid JSON.`,
        },
        {
          role: 'user',
          content: `Research question: ${question}\n\nReport template:\n${template.slice(0, 6000)}\n\nLanguage: ${langName}`,
        },
      ],
    }), 'planResearch', 30000);

    trackUsage(PLANNING_MODEL, response.usage);

    try {
      const plan = JSON.parse(response.choices[0].message.content);
      // Validate and provide defaults
      return {
        sections: plan.sections || ['Introduction', 'Analysis', 'Conclusion'],
        search_queries: (plan.search_queries || []).slice(0, MAX_SEARCH_TURNS),
        key_questions: (plan.key_questions || []).slice(0, 15),
      };
    } catch (parseErr) {
      console.log(`[Researcher-OpenAI] Plan parse failed, using fallback plan`);
      return buildFallbackPlan(question, depthCfg);
    }
  }

  async function searchAndExtract(client, query, mainQuestion, langName) {
    const response = await withTimeout(client.responses.create({
      model: RESEARCH_MODEL,
      instructions: `You are a research data extractor with source quality assessment. Search the web for the given query and extract ALL relevant data points.

For each piece of information found, record:
- [A/B/C] Source tier grade
- The specific data point (number, fact, quote, finding)
- The source name and URL in markdown link format: [Source Name](https://url)
- The date/year of the data

SOURCE TIER GRADING (mandatory for every data point):
- [A] Primary/institutional: peer-reviewed journals (PubMed, Nature, Lancet, JAMA), government agencies (FDA, WHO, EU Commission, OECD), official statistics (Eurostat, FRED, World Bank), established research institutions
- [B] High-quality secondary: major news outlets (Reuters, Bloomberg, FT, NYT), reputable industry reports (McKinsey, Gartner, BCG), professional associations (AAFCO, IEEE), established encyclopedias (Britannica)
- [C] Low confidence: blogs, SEO content sites, marketing pages, unknown domains, opinion pieces, user-generated content, social media

Be thorough. Extract statistics, names, dates, quotes, percentages. Do NOT summarize or editorialize -- just extract raw data with tier grades.

Format as a bullet list. Include the tier grade AND URL for EVERY data point.

If searching in a non-English context, also search for ${langName} language sources.`,
      input: [
        {
          role: 'user',
          content: `Search query: ${query}\n\nThis is part of a larger research project about: ${mainQuestion}\n\nSearch the web and extract all relevant data points with sources.`,
        },
      ],
      tools: [{ type: 'web_search_preview' }],
      max_output_tokens: 4000,
    }), `search:${query.slice(0, 40)}`, 30000);

    const text = extractResponseText(response);
    // Count web search tool calls in the response for accurate cost tracking
    const webSearchCalls = response.output
      ? response.output.filter(item => item.type === 'web_search_call').length
      : 0;
    return { text, usage: response.usage, webSearchCalls };
  }

  /**
   * Execute a batch of search queries and append results to ctx.scratchpad.
   */
  async function executeSearchRound(client, queries, ctx, batchSize, writeReserveMs, hasTime, trackUsage, roundNum, label = null) {
    let batchNum = 0;
    for (let batchStart = 0; batchStart < queries.length; ) {
      if (pastDeadline() || !hasTime(writeReserveMs)) {
        console.log(`[Researcher-OpenAI] Time limit approaching after ${ctx.searchCount} searches, stopping round ${roundNum}`);
        break;
      }

      // Check memory pressure and reduce batch size if needed
      const effectiveBatchSize = checkMemoryPressure(batchSize);
      if (effectiveBatchSize < batchSize) {
        console.log(`[Researcher-OpenAI] Reduced batch size from ${batchSize} to ${effectiveBatchSize} due to memory pressure`);
      }

      const batch = queries.slice(batchStart, batchStart + effectiveBatchSize);
      batchStart += effectiveBatchSize;
      batchNum++;
      const batchLabel = label || `R${roundNum}`;
      console.log(`[Researcher-OpenAI] ${batchLabel} Batch ${batchNum}: searching ${batch.length} queries in parallel...`);

      const results = await Promise.allSettled(
        batch.map((query, idx) => searchAndExtract(client, query, ctx.question, langName)
          .then(findings => ({ query, findings, idx: batchStart + idx }))
          .catch(err => ({ query, error: err.message, idx: batchStart + idx }))
        )
      );

      for (const result of results) {
        const r = result.value || result.reason;
        ctx.allQueriesRun.add(r.query);
        if (r.error) {
          console.log(`[Researcher-OpenAI]   [${batchLabel}] "${r.query.slice(0, 50)}..." -> FAILED: ${r.error}`);
          continue;
        }
        if (r.findings?.text && r.findings.text.trim().length > 50) {
          ctx.scratchpad += `\n---\n### Search: "${r.query}"\n${r.findings.text}\n`;
          ctx.searchCount++;
          trackUsage(RESEARCH_MODEL, r.findings.usage);
          // Track web search tool call costs
          const wsCalls = r.findings.webSearchCalls || 0;
          if (wsCalls > 0) {
            totalWebSearchCalls += wsCalls;
            totalCost += wsCalls * WEB_SEARCH_COST_PER_CALL;
          }
          console.log(`[Researcher-OpenAI]   [${batchLabel}] "${r.query.slice(0, 50)}..." -> ${r.findings.text.split(/\s+/).length} words (${wsCalls} web searches)`);
        } else {
          console.log(`[Researcher-OpenAI]   [${batchLabel}] "${r.query.slice(0, 50)}..." -> no useful data`);
          if (r.findings?.usage) trackUsage(RESEARCH_MODEL, r.findings.usage);
          // Still track web search costs even if no useful data returned
          const wsCalls = r.findings?.webSearchCalls || 0;
          if (wsCalls > 0) {
            totalWebSearchCalls += wsCalls;
            totalCost += wsCalls * WEB_SEARCH_COST_PER_CALL;
          }
        }
      }
      const wordCount = ctx.scratchpad.split(/\s+/).length;
      console.log(`[Researcher-OpenAI] Scratchpad: ${wordCount} words after batch`);
    }
  }

  /**
   * Assess research coverage: which sections are strong, weak, or missing?
   * Returns additional targeted queries for weak sections.
   */
  async function assessCoverage(client, ctx, sections, trackUsage) {
    // Take a representative sample of the scratchpad (last 30k chars) to stay within context limits
    const scratchpadSample = ctx.scratchpad.length > 30000
      ? ctx.scratchpad.slice(0, 5000) + '\n...[truncated]...\n' + ctx.scratchpad.slice(-25000)
      : ctx.scratchpad;

    const response = await withTimeout(client.chat.completions.create({
      model: ASSESSMENT_MODEL,
      temperature: 0.2,
      max_completion_tokens: 4000,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: `You are a research quality assessor. Given a research scratchpad and planned sections, assess coverage quality and identify gaps.

For each section, determine if evidence is:
- "strong": 3+ data points from A or B tier sources with specific numbers/dates
- "weak": fewer than 3 data points, or mostly C-tier sources, or vague claims without specifics
- "missing": no relevant data found

Also identify:
- Low-quality domains that appeared frequently (SEO blogs, content farms, unknown sites)
- Specific gaps: what data is missing that a targeted search could find?
- Generate 5-15 additional search queries targeting the weak/missing sections

For additional queries, be SPECIFIC and TARGET authoritative sources:
- Use site: prefix for known authoritative domains
- Include year markers (2024, 2025, 2026)
- Target the specific data gap (not generic queries)

Output JSON with:
{
  "sectionCoverage": { "Section Name": "strong"|"weak"|"missing", ... },
  "rejectedDomains": ["lowquality.com", ...],
  "gaps": ["description of what is missing", ...],
  "additionalQueries": ["specific search query", ...],
  "overallAssessment": "brief summary"
}`,
        },
        {
          role: 'user',
          content: `# Research Question\n${ctx.question}\n\n# Planned Sections\n${sections.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\n# Research Scratchpad (collected data):\n${scratchpadSample}\n\n# Previously rejected domains:\n${ctx.rejectedSources.join(', ') || '(none yet)'}\n\nAssess coverage and generate targeted queries for gaps.`,
        },
      ],
    }), 'assessCoverage', 30000);

    trackUsage(ASSESSMENT_MODEL, response.usage);

    try {
      const assessment = JSON.parse(response.choices[0].message.content);
      console.log(`[Researcher-OpenAI] Assessment: ${assessment.overallAssessment || '(no summary)'}`);
      return assessment;
    } catch (parseErr) {
      console.log(`[Researcher-OpenAI] Assessment parse failed: ${parseErr.message}`);
      return { sectionCoverage: {}, rejectedDomains: [], gaps: [], additionalQueries: [] };
    }
  }

  /**
   * Paragraph-level self-review: check each paragraph for unsourced claims,
   * vague language, repetition, and filler. Fix issues before returning.
   */
  async function reviewAndFixSection(client, sectionText, previousSections, sectionName, ctx, trackUsage) {
    console.log(`[Researcher-OpenAI]     Reviewing "${sectionName}" at paragraph level...`);

    // Step 1: Identify issues (cheap, mini model)
    const reviewResponse = await withTimeout(client.chat.completions.create({
      model: ASSESSMENT_MODEL,
      temperature: 0.1,
      max_completion_tokens: 2000,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: `You are a research report quality reviewer. Analyze the given section paragraph by paragraph.

For each paragraph, check:
1. Is every factual claim sourced with a markdown link? If not, flag which claim needs a source.
2. Are there vague phrases like "experts say", "studies show", "research indicates", "it is widely believed"? Flag with the exact phrase and what specific source/data should replace it.
3. Is there content repeated from previous sections? Flag with what is repeated.
4. Is the paragraph adding analytical value or is it filler/padding? Flag filler paragraphs.

Output JSON:
{
  "issues": [
    {
      "paragraph_start": "first 10 words of the paragraph...",
      "issue_type": "unsourced_claim"|"vague_language"|"repetition"|"filler",
      "description": "what is wrong",
      "fix_instruction": "specific instruction for how to fix it"
    }
  ],
  "issue_count": N,
  "quality_score": 1-10
}

If the section is high quality with no issues, return {"issues": [], "issue_count": 0, "quality_score": 9}.
Be strict. Academic-grade sourcing is expected.`,
        },
        {
          role: 'user',
          content: `# Section: "${sectionName}"\n\n${sectionText}\n\n# Previous sections (for repetition check, last 2000 chars):\n...${previousSections.slice(-2000)}`,
        },
      ],
    }), `review:${sectionName}`, 30000);

    trackUsage(ASSESSMENT_MODEL, reviewResponse.usage);

    let review;
    try {
      review = JSON.parse(reviewResponse.choices[0].message.content);
    } catch {
      console.log(`[Researcher-OpenAI]     Review parse failed, keeping section as-is`);
      return sectionText;
    }

    if (!review.issues || review.issues.length === 0) {
      console.log(`[Researcher-OpenAI]     Section "${sectionName}" passed review (score: ${review.quality_score || '?'}/10)`);
      return sectionText;
    }

    console.log(`[Researcher-OpenAI]     Found ${review.issues.length} issues (score: ${review.quality_score || '?'}/10), fixing...`);

    // Step 2: Fix issues (quality model)
    const fixResponse = await withTimeout(client.chat.completions.create({
      model: WRITING_MODEL,
      temperature: 0.15,
      max_completion_tokens: MAX_WRITE_TOKENS,
      messages: [
        {
          role: 'system',
          content: `You are a research report editor. Fix the identified issues in the section below. Rules:
- Remove or rewrite vague language ("experts say" -> cite the specific expert/study, or remove the claim)
- Remove unsourced claims UNLESS you can cite them from the research context provided
- Remove filler paragraphs entirely
- Remove repeated content (brief cross-reference is OK: "as discussed in [Section]")
- Preserve all well-sourced content unchanged
- Do NOT add new information — only fix or remove problematic content
- Return the complete fixed section text (not just the changes)`,
        },
        {
          role: 'user',
          content: `# Section to fix: "${sectionName}"\n\n${sectionText}\n\n# Issues found:\n${JSON.stringify(review.issues, null, 2)}\n\n# Research context (for finding missing sources):\n${ctx.scratchpad.slice(-8000)}`,
        },
      ],
    }), `fixSection:${sectionName}`, 30000);

    trackUsage(WRITING_MODEL, fixResponse.usage);

    const fixedText = fixResponse.choices[0]?.message?.content || '';
    if (fixedText.trim().length > sectionText.trim().length * 0.5) {
      console.log(`[Researcher-OpenAI]     Fixed ${review.issues.length} issues in "${sectionName}"`);
      return fixedText;
    }

    console.log(`[Researcher-OpenAI]     Fix output too short, keeping original`);
    return sectionText;
  }

  async function writeSectionDraft(client, opts) {
    const {
      sectionName, evidence, question, template, langName, depthConfig: depthCfg,
      allSections, previousSections, isIntro = false, isConclusion = false,
      fullReportSoFar = '', ctx: researchCtx,
    } = opts;

    let userContent;

    // Build context summary from research assessments
    const contextSummary = researchCtx?.assessments?.length > 0
      ? `\n\n## Research Assessment Notes:\n${researchCtx.assessments.map(a => a.overallAssessment || '').filter(Boolean).join('\n')}\nSection coverage: ${JSON.stringify(researchCtx.sectionCoverage || {})}`
      : '';

    if (isIntro) {
      userContent = `# Research Question\n${question}\n\n# Report Structure\n${allSections.map((s, i) => `${i + 1}. ${s}`).join('\n')}${contextSummary}\n\n---\n\nWrite the report title (as # heading), executive summary, and methodology section. Set up the report structure. This will be followed by detailed analysis sections.\n\nAim for 300-600 words for this introduction.`;
    } else if (isConclusion) {
      userContent = `# Research Question\n${question}\n\n# Report written so far (last 5000 chars):\n...${fullReportSoFar.slice(-5000)}${contextSummary}\n\n---\n\nWrite the conclusion section that synthesizes findings across all sections, followed by a complete source list. The source list should contain EVERY URL cited in the report above, organized by category.\n\nAim for 500-1000 words.`;
    } else {
      // Format evidence pack for this section -- prioritize by source tier then relevance
      const tierOrder = { tier1_institutional: 0, tier2_industry: 1, tier3_news: 2, tier4_opinion: 3 };
      const claimsText = (evidence.claims || [])
        .sort((a, b) => {
          const tierDiff = (tierOrder[a.source_tier] || 3) - (tierOrder[b.source_tier] || 3);
          if (tierDiff !== 0) return tierDiff;
          return (b.relevance || 0) - (a.relevance || 0);
        })
        .map(c => `- [${(c.source_tier || 'unknown').replace('tier1_', 'A:').replace('tier2_', 'B:').replace('tier3_', 'C:').replace('tier4_', 'D:')}] ${c.text} [${c.source_name}](${c.source_url}) (${c.year || 'n/a'})`)
        .join('\n');

      const statsText = (evidence.key_stats || [])
        .map(s => `- ${s.stat} [${s.source_name}](${s.source_url}) (${s.year || 'n/a'})`)
        .join('\n');

      const contradictionsText = (evidence.contradictions || [])
        .map(c => `- CONTRADICTION: ${c.claim_a} vs ${c.claim_b}`)
        .join('\n');

      userContent = `# Section to write: "${sectionName}"
# Research Question: ${question}

## Evidence for this section:

### Claims (${(evidence.claims || []).length} items, sorted by relevance):
${claimsText || '(no claims extracted)'}

### Key Statistics:
${statsText || '(no statistics)'}

### Contradictions to address:
${contradictionsText || '(none found)'}
${contextSummary}

## Context (last part of report written so far):
...${previousSections}

---

Write ONLY the section "${sectionName}" as markdown (starting with ## heading).

SOURCE QUALITY RULES:
- Claims marked [A:] are from institutional/academic sources -- these are PRIMARY evidence. Build the section around them.
- Claims marked [B:] are from industry/major news -- use as SUPPORTING evidence.
- Claims marked [C:] or [D:] are low-confidence -- use ONLY if no A/B source covers the topic, and mark as "unverified" or "according to [source]".
- If A/B sources contradict C/D sources, always prioritize A/B.
- NEVER present a C/D source claim as established fact.

Cite every claim with its markdown link. Address contradictions where they exist. Build tables where data supports it.

Aim for 500-2000 words depending on evidence density.

CRITICAL: Do NOT repeat information already covered in previous sections. If a data point was mentioned earlier, reference it briefly ("as noted in Section X") but do NOT restate it. Each section must contain UNIQUE analysis and data points. Repetition is a critical failure.

PREVIOUS SECTIONS ALREADY WRITTEN (do not repeat this content):
${previousSections.slice(-2000)}`;
    }

    const response = await withTimeout(client.chat.completions.create({
      model: WRITING_MODEL,
      temperature: 0.2,
      max_completion_tokens: MAX_WRITE_TOKENS,
      messages: [
        {
          role: 'system',
          content: buildWritingSystemPrompt(template, langName, depthCfg),
        },
        {
          role: 'user',
          content: userContent,
        },
      ],
    }), `writeDraft:${sectionName}`, 60000);

    return { text: response.choices[0]?.message?.content || '', usage: response.usage };
  }

  async function verifyReport(client, report, scratchpad) {
    const response = await withTimeout(client.chat.completions.create({
      model: VERIFICATION_MODEL,
      temperature: 0,
      max_completion_tokens: MAX_WRITE_TOKENS,
      messages: [
        {
          role: 'system',
          content: `You are a source verification editor. Review the research report and:

1. Check that EVERY factual claim has a source citation in markdown link format [Name](URL)
2. Check that all cited URLs actually appear in the research scratchpad
3. Remove any claims that lack sources and cannot be verified from the scratchpad
4. Add missing citations where the data exists in the scratchpad but was not cited
5. Ensure the source list at the end is complete

Return the full corrected report. Make minimal changes -- only fix citation issues.`,
        },
        {
          role: 'user',
          content: `# Report to verify:\n${report}\n\n# Source scratchpad:\n${scratchpad.slice(0, 100_000)}`,
        },
      ],
    }), 'verifyReport', 60000);

    trackUsage(VERIFICATION_MODEL, response.usage);
    return { text: response.choices[0]?.message?.content || '', usage: response.usage };
  }
}


// ====================================================================
// Utility functions
// ====================================================================

function fillTemplate(templateContent, question, language) {
  return templateContent
    .replace(/\{\{TOPIC\}\}/g, question)
    .replace(/\{\{LANGUAGE\}\}/g, language === 'en' ? 'English' : language === 'sv' ? 'Swedish' : language)
    .replace(/\{\{SEGMENT\}\}/g, 'general audience')
    .replace(/\{\{MARKET\}\}/g, 'global')
    .replace(/\{\{COUNTRY\}\}/g, 'global')
    .replace(/\{\{TIMEFRAME\}\}/g, '1-3 years')
    .replace(/\{\{PROJECT_DIR\}\}/g, '/tmp/aurum-research');
}

function getDepthConfig(depth) {
  switch (depth) {
    case 'deep':
      return { queryCount: [30, 45], minWords: 8000, maxWords: 18000 };
    case 'quick':
      return { queryCount: [10, 15], minWords: 2000, maxWords: 4000 };
    default: // standard
      return { queryCount: [20, 30], minWords: 5000, maxWords: 10000 };
  }
}

function buildFallbackPlan(question, depthCfg) {
  const base = question.slice(0, 100);
  return {
    sections: [
      'Executive Summary',
      'Background and Context',
      'Current State Analysis',
      'Key Findings',
      'Trends and Drivers',
      'Scenarios and Outlook',
      'Recommendations',
      'Sources',
    ],
    search_queries: [
      `${base} overview 2025 2026`,
      `${base} market size statistics`,
      `${base} latest news developments`,
      `${base} trends forecast`,
      `${base} analysis report`,
      `${base} challenges risks`,
      `${base} opportunities growth`,
      `${base} expert opinion`,
      `${base} data statistics numbers`,
      `${base} comparison alternatives`,
      `${base} case study example`,
      `${base} future outlook predictions`,
      `${base} pros cons advantages disadvantages`,
      `${base} industry report 2025`,
      `${base} research findings study`,
      `site:statista.com ${base}`,
      `site:reuters.com ${base}`,
      `site:bloomberg.com ${base}`,
      `${base} counter argument criticism`,
      `${base} impact consequences effects`,
    ].slice(0, depthCfg.queryCount[1]),
    key_questions: [
      `What is the current state of ${base}?`,
      `What are the main trends driving ${base}?`,
      `What are the key risks and challenges?`,
      `What do experts predict for the future?`,
      `What are the counter-arguments or criticisms?`,
    ],
  };
}

function buildWritingSystemPrompt(template, langName, depthCfg) {
  return `You are Aurum, an elite research intelligence engine. Write a comprehensive, publication-ready research report.

LANGUAGE: Write the entire report in ${langName}.
TARGET LENGTH: ${depthCfg.minWords}-${depthCfg.maxWords} words.

REPORT TEMPLATE AND METHODOLOGY:
${template.slice(0, 8000)}

CRITICAL RULES:
- Use ONLY data from the research scratchpad provided. Do NOT fabricate any data, statistics, or sources.
- EVERY factual claim MUST have a source citation as a markdown link: [Source Name](https://url)
- If the scratchpad does not contain data for a section, explicitly state that data was not found rather than inventing it.
- Use Sherman Kent probability language where appropriate: Almost certain (>93%), Probable (63-87%), Roughly even (40-60%), Unlikely (13-37%), Remote (<7%).
- Include specific numbers, names, dates, and URLs.
- Build tables where the template calls for them.
- The report must be self-contained and publication-ready.
- Start with a top-level markdown heading.
- End with a complete source list organized by category.

OUTPUT FORMAT: Write the complete report as a single markdown document.`;
}

function extractResponseText(response) {
  let text = '';
  if (!response.output) return text;

  for (const block of response.output) {
    if (block.type === 'message' && block.content) {
      for (const c of block.content) {
        if (c.type === 'output_text') {
          text += c.text + '\n';
        }
      }
    }
  }
  return text;
}

function buildSystemPrompt(templateContent, language, depth) {
  // Kept for backward compatibility, though no longer used in main flow
  const langInstruction = language === 'en'
    ? 'Write the entire report in English.'
    : language === 'sv'
      ? 'Write the entire report in Swedish.'
      : `Write the entire report in ${language}.`;

  const depthInstruction = depth === 'deep'
    ? 'This is a deep research request. Be extremely thorough -- aim for 8000+ words, 100+ sources.'
    : depth === 'quick'
      ? 'This is a quick research request. Be concise -- aim for 2000-3000 words with key findings.'
      : 'This is a standard research request. Aim for 5000+ words with comprehensive analysis.';

  return `You are Aurum, an elite research intelligence engine. Produce a comprehensive, publication-ready research report.

${langInstruction}
${depthInstruction}

METHODOLOGY:
${templateContent}

CRITICAL RULES:
- Search the web extensively for current, real data. NEVER fabricate sources or statistics.
- Every claim must be sourced with a URL in markdown link format: [Source Name](https://url)
- Use Sherman Kent probability scale: Almost certain (>93%), Probable (63-87%), Roughly even (40-60%), Unlikely (13-37%), Remote (<7%)
- Include specific numbers, names, dates
- Build the report section by section following the template structure
- The report must be self-contained

OUTPUT FORMAT:
Write the complete report as a single markdown document. Start with a top-level heading.`;
}

function buildUserPrompt(question, language) {
  return `Research the following question and produce a complete intelligence report:\n\n${question}\n\nUse web search to gather current data. Write the full report in markdown format.`;
}

function cleanReport(text) {
  let cleaned = text
    .replace(/^(Here'?s? (?:is )?(?:the|my|your) (?:complete |full |final )?(?:report|analysis|intelligence report)[:\s]*\n*)/i, '')
    .replace(/\n{4,}/g, '\n\n\n');
  return cleaned;
}
