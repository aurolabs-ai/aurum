// PROPRIETARY -- Aurolabs AB. All rights reserved. Not for distribution.
//
// Agentic researcher v2: iterative agentic research.
//
// Architecture: understand() -> [search(5) -> analyze(findings) -> update state]* -> write() -> review()
//
// Key design principles:
// 1. The model drives its own search strategy -- reactive, not pre-planned
// 2. Each round: 3-5 searches, then full analysis of what was found
// 3. State tracking: checklist of what's known vs unknown, data points collected, gaps remaining
// 4. The model SEES its accumulated knowledge when deciding next searches
// 5. Searches include free queries, site:-targeted queries, prediction markets, sentiment
// 6. Continue until: all checklist items covered OR time limit OR diminishing returns

import OpenAI from 'openai';
import { loadTemplate } from './template-matcher.js';

const RESEARCH_MODEL = 'gpt-5.4-mini';
const WRITING_MODEL = 'gpt-5.4';

const PRICING = {
  'gpt-5.4':      { in: 2.5,  out: 15.0 },
  'gpt-5.4-mini': { in: 0.75, out: 4.5  },
};

const WEB_SEARCH_COST_PER_CALL = 0.025;

const MANDATORY_SOURCES = [
  { name: 'Polymarket', prefix: 'site:polymarket.com' },
  { name: 'Kalshi', prefix: 'site:kalshi.com' },
  { name: 'Reddit', prefix: 'site:reddit.com' },
  { name: 'Yahoo Finance', prefix: 'site:finance.yahoo.com' },
  { name: 'FRED', prefix: 'site:fred.stlouisfed.org' },
  { name: 'Eurostat', prefix: 'site:ec.europa.eu/eurostat' },
  { name: 'IMF', prefix: 'site:imf.org' },
  { name: 'IEA', prefix: 'site:iea.org' },
];

/**
 * Agentic research v2: iterative agentic knowledge building.
 *
 * Instead of: plan 30 queries -> execute blindly -> write
 * This does:  understand -> search(5) -> analyze -> search(5) -> ... -> write -> review
 *
 * The model sees ALL its accumulated findings and decides what to search next.
 */
export async function researchAgent(question, openaiKey, options = {}) {
  const {
    templateName,
    language = 'en',
    depth = 'standard',
    timeLimitMs = 30 * 60 * 1000,
    maxSearchRounds = 20,
    costTracker = {},
  } = options;

  const client = new OpenAI({ apiKey: openaiKey });
  const langName = language === 'en' ? 'English' : language === 'sv' ? 'Swedish' : language;
  const startTime = Date.now();

  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCost = 0;
  let totalWebSearchCalls = 0;
  let totalSearchRounds = 0;

  function elapsed() { return Date.now() - startTime; }
  function timeLeft() { return timeLimitMs - elapsed(); }
  function hasTime(reserveMs = 180_000) { return timeLeft() > reserveMs; }

  function trackUsage(model, usage) {
    if (!usage) return;
    const tokIn = usage.input_tokens || usage.prompt_tokens || 0;
    const tokOut = usage.output_tokens || usage.completion_tokens || 0;
    const p = PRICING[model] || PRICING['gpt-5.4-mini'];
    const cost = (tokIn * p.in / 1_000_000) + (tokOut * p.out / 1_000_000);
    totalTokensIn += tokIn;
    totalTokensOut += tokOut;
    totalCost += cost;
    return { tokIn, tokOut, cost };
  }

  // Load template for writing phase
  let templateContent = '';
  try {
    templateContent = await loadTemplate(templateName || 'scenario-analysis');
  } catch {
    console.log('[Agent] Template load failed, proceeding without template');
  }

  console.log(`[Agent] Starting agentic research v2: "${question.slice(0, 80)}..."`);
  console.log(`[Agent] Models: ${RESEARCH_MODEL} (research), ${WRITING_MODEL} (writing)`);
  console.log(`[Agent] Time limit: ${Math.round(timeLimitMs / 60000)} min, max ${maxSearchRounds} rounds`);

  // ========== PHASE 1: Understand the Question ==========
  console.log(`[Agent] === PHASE 1: Understanding question ===`);

  const understandResponse = await client.responses.create({
    model: RESEARCH_MODEL,
    instructions: `You are an intelligence analyst planning a research operation.
Classify the question and create a research plan. Be specific and actionable.

Respond in EXACTLY this JSON format (no markdown fencing):
{
  "classification": "policy|market-research|due-diligence|geopolitical|technology|sector-analysis|risk-assessment",
  "topic_keywords": ["keyword1", "keyword2", ...],
  "checklist": [
    {"id": 1, "item": "What specific thing needs to be answered", "priority": "must|should|nice", "status": "pending"},
    ...
  ],
  "data_sources_needed": ["financial_data", "news", "institutional", "sentiment", "prediction_markets", "company_filings", "government_data"],
  "initial_queries": [
    "first free-form search query",
    "second query",
    "site:specific-source.com targeted query",
    "another query",
    "site:polymarket.com topic prediction"
  ],
  "mandatory_source_queries": [
    "site:polymarket.com topic",
    "site:reddit.com topic discussion"
  ]
}

Create 12-20 checklist items covering all aspects that need answering.
Generate 5-8 initial search queries (mix of free and site:-targeted).
Include mandatory source queries for prediction markets and sentiment.`,
    input: [{ role: 'user', content: `Research question: "${question}"\nLanguage: ${langName}` }],
    max_output_tokens: 4000,
  });

  trackUsage(RESEARCH_MODEL, understandResponse.usage);

  let researchPlan;
  const planText = extractText(understandResponse);
  try {
    // Try to parse JSON, stripping markdown fences if present
    const cleaned = planText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    researchPlan = JSON.parse(cleaned);
  } catch {
    console.log('[Agent] Failed to parse research plan, using fallback');
    researchPlan = buildFallbackPlan(question);
  }

  const checklist = researchPlan.checklist || [];
  console.log(`[Agent] Classification: ${researchPlan.classification}`);
  console.log(`[Agent] Checklist: ${checklist.length} items`);
  console.log(`[Agent] Initial queries: ${(researchPlan.initial_queries || []).length}`);

  // ========== PHASE 2: Iterative Research Loop ==========
  console.log(`[Agent] === PHASE 2: Iterative research ===`);

  // Accumulated state -- the model sees this growing each round
  const state = {
    dataPoints: [],        // { fact, source_url, source_name, confidence, round }
    sourcesSearched: [],   // all queries run
    domainsHit: new Set(),
    mandatorySourcesCovered: new Set(),
    roundSummaries: [],    // summary of each round's findings
    contradictions: [],    // conflicting data points
    gapsClosed: [],        // checklist items resolved
    gapsRemaining: checklist.map(c => c.item),
  };

  // Round 1 uses initial queries from the plan
  let nextQueries = [
    ...(researchPlan.initial_queries || []),
    ...(researchPlan.mandatory_source_queries || []),
  ].slice(0, 8);

  for (let round = 1; round <= maxSearchRounds; round++) {
    if (!hasTime()) {
      console.log(`[Agent] Time limit approaching, stopping at round ${round}`);
      break;
    }

    totalSearchRounds = round;

    // --- SEARCH: Execute queries (model does the searching) ---
    const searchPrompt = buildSearchPrompt(question, langName, nextQueries, round, state);

    let searchResponse;
    try {
      searchResponse = await client.responses.create({
        model: RESEARCH_MODEL,
        instructions: `You are a research analyst executing web searches. For EACH query provided, search the web and extract specific data points.

For every piece of information found:
- Extract the SPECIFIC data: numbers, dates, names, percentages, amounts
- Note the exact source URL
- Grade confidence: A (primary/institutional), B (quality news/research), C (blog/opinion)
- Flag any contradictions with previously known data

You MUST search for ALL queries provided. Do not skip any.
After all searches, provide a structured summary of what you found and what's still missing.`,
        input: [{ role: 'user', content: searchPrompt }],
        tools: [{ type: 'web_search_preview' }],
        max_output_tokens: 12000,
      });
    } catch (err) {
      console.log(`[Agent] Search round ${round} failed: ${err.message}`);
      if (round === 1) throw err;
      break;
    }

    trackUsage(RESEARCH_MODEL, searchResponse.usage);

    // Count web search calls
    const webCalls = searchResponse.output
      ? searchResponse.output.filter(item => item.type === 'web_search_call').length
      : 0;
    totalWebSearchCalls += webCalls;
    totalCost += webCalls * WEB_SEARCH_COST_PER_CALL;

    const roundFindings = extractText(searchResponse);
    state.sourcesSearched.push(...nextQueries);

    // Extract domains from findings
    const urlMatches = roundFindings.match(/https?:\/\/[^\s\)]+/g) || [];
    for (const url of urlMatches) {
      try { state.domainsHit.add(new URL(url).hostname); } catch {}
    }

    // Track mandatory source coverage
    for (const src of MANDATORY_SOURCES) {
      const prefix = src.prefix.replace('site:', '');
      if (urlMatches.some(u => u.includes(prefix))) {
        state.mandatorySourcesCovered.add(src.name);
      }
    }

    const roundWords = roundFindings.split(/\s+/).length;
    state.roundSummaries.push({
      round,
      webCalls,
      words: roundWords,
      findings: roundFindings,
    });

    console.log(`[Agent] Round ${round}: ${webCalls} web searches, ${roundWords} words, ${state.domainsHit.size} unique domains`);

    // --- ANALYZE: Assess findings and decide next actions ---
    if (!hasTime(120_000)) {
      console.log(`[Agent] Insufficient time for analysis, proceeding to write`);
      break;
    }

    const analyzePrompt = buildAnalyzePrompt(question, langName, round, state, checklist);

    let analyzeResponse;
    try {
      analyzeResponse = await client.responses.create({
        model: RESEARCH_MODEL,
        instructions: `You are a research director analyzing findings and steering the next round of research.

Your job:
1. Extract and list all SPECIFIC data points found (numbers, dates, names, URLs)
2. Update the checklist: which items are now answered? Which remain?
3. Identify contradictions between sources
4. Identify the BIGGEST GAPS: what critical information is still missing?
5. Generate 3-5 NEW search queries to fill the most important gaps
6. Decide if research is sufficient or more rounds are needed

Respond in EXACTLY this JSON format (no markdown fencing):
{
  "new_data_points": [
    {"fact": "specific data point", "source": "source name", "url": "https://...", "confidence": "A|B|C"}
  ],
  "checklist_updates": [
    {"id": 1, "status": "done|partial|pending", "evidence": "what was found"}
  ],
  "contradictions": [
    {"claim1": "...", "source1": "...", "claim2": "...", "source2": "...", "resolution": "..."}
  ],
  "gaps": ["gap 1 description", "gap 2", ...],
  "next_queries": [
    "specific search query 1",
    "site:specific-source.com targeted query 2",
    ...
  ],
  "mandatory_sources_missing": ["Polymarket", "Reddit", ...],
  "ready_to_write": false,
  "reasoning": "Why more research is/isn't needed"
}

CRITICAL: Generate at least 3 next_queries unless ready_to_write is true.
Queries should target GAPS, not repeat what's already known.
Mix free queries with site:-targeted queries for specific sources.`,
        input: [{ role: 'user', content: analyzePrompt }],
        max_output_tokens: 6000,
      });
    } catch (err) {
      console.log(`[Agent] Analysis round ${round} failed: ${err.message}`);
      break;
    }

    trackUsage(RESEARCH_MODEL, analyzeResponse.usage);

    const analysisText = extractText(analyzeResponse);
    let analysis;
    try {
      const cleaned = analysisText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      analysis = JSON.parse(cleaned);
    } catch {
      console.log(`[Agent] Failed to parse analysis JSON, extracting queries manually`);
      analysis = extractFallbackAnalysis(analysisText);
    }

    // Update state from analysis
    if (analysis.new_data_points) {
      for (const dp of analysis.new_data_points) {
        state.dataPoints.push({ ...dp, round });
      }
    }
    if (analysis.checklist_updates) {
      for (const update of analysis.checklist_updates) {
        const item = checklist.find(c => c.id === update.id);
        if (item) item.status = update.status;
      }
      state.gapsRemaining = checklist.filter(c => c.status !== 'done').map(c => c.item);
      state.gapsClosed = checklist.filter(c => c.status === 'done').map(c => c.item);
    }
    if (analysis.contradictions) {
      state.contradictions.push(...analysis.contradictions);
    }

    const checklistDone = checklist.filter(c => c.status === 'done').length;
    const checklistTotal = checklist.length;
    const uncoveredMandatory = MANDATORY_SOURCES
      .map(s => s.name)
      .filter(name => !state.mandatorySourcesCovered.has(name));

    console.log(`[Agent] Analysis: ${(analysis.new_data_points || []).length} new data points, ${checklistDone}/${checklistTotal} checklist done, ${state.dataPoints.length} total data points`);
    console.log(`[Agent] Mandatory sources covered: ${state.mandatorySourcesCovered.size}/${MANDATORY_SOURCES.length} (missing: ${uncoveredMandatory.join(', ') || 'none'})`);

    // Decide whether to continue
    if (analysis.ready_to_write && checklistDone >= checklistTotal * 0.7 && round >= 3) {
      console.log(`[Agent] Agent says ready to write after ${round} rounds (${checklistDone}/${checklistTotal} items done)`);
      break;
    }

    // Force-add queries for uncovered mandatory sources (if any remain and we haven't tried them)
    let mandatoryQueries = [];
    if (uncoveredMandatory.length > 0 && round <= maxSearchRounds - 2) {
      const mandatorySrc = MANDATORY_SOURCES.filter(s => uncoveredMandatory.includes(s.name));
      mandatoryQueries = mandatorySrc
        .filter(s => !state.sourcesSearched.some(q => q.includes(s.prefix)))
        .map(s => `${s.prefix} ${researchPlan.topic_keywords ? researchPlan.topic_keywords.slice(0, 3).join(' ') : question.slice(0, 40)}`)
        .slice(0, 2);
    }

    // Set next queries
    nextQueries = [
      ...(analysis.next_queries || []),
      ...mandatoryQueries,
    ].slice(0, 6);

    if (nextQueries.length === 0) {
      console.log(`[Agent] No more queries to run, proceeding to write`);
      break;
    }

    // Diminishing returns check
    if (round >= 5 && (analysis.new_data_points || []).length < 2) {
      console.log(`[Agent] Diminishing returns (${(analysis.new_data_points || []).length} new points), proceeding to write`);
      break;
    }
  }

  const totalDataPoints = state.dataPoints.length;
  const totalDomains = state.domainsHit.size;
  console.log(`[Agent] Research complete: ${totalSearchRounds} rounds, ${totalWebSearchCalls} web searches, ${totalDataPoints} data points, ${totalDomains} domains`);

  // ========== PHASE 3: Write Report ==========
  console.log(`[Agent] === PHASE 3: Writing report ===`);

  if (!hasTime(60_000)) {
    console.log(`[Agent] WARNING: Very limited time for writing`);
  }

  const writePrompt = buildWritePrompt(question, langName, state, checklist, templateContent);

  let report;
  try {
    const writeResponse = await client.responses.create({
      model: WRITING_MODEL,
      instructions: `You are a senior intelligence analyst writing a comprehensive research report.
Write in ${langName}. Be thorough, specific, and source every claim.

QUALITY STANDARDS:
- Every factual claim MUST have a source URL in markdown link format: [Source](URL)
- Use specific data: numbers, dates, percentages, named entities -- never "several" or "various"
- Use Sherman Kent probability language: Almost certain (93%+), Probable (75-85%), Roughly even (45-55%), Unlikely (15-25%), Remote (<7%)
- Surface contradictions explicitly -- do not present a single narrative
- Include tables for structured data (deals, metrics, comparisons)
- Minimum 8 sections, each with substantive analysis (500+ words)
- Begin with a Methodology Note (data collection date, sources searched, confidence levels, limitations)
- End with a complete Source List organized by category

DO NOT use generic filler. Every sentence should carry specific information.
DO NOT write "in conclusion" or "as we have seen" or other transitional fluff.
DO NOT use emojis in the report body.`,
      input: [{ role: 'user', content: writePrompt }],
      tools: [{ type: 'web_search_preview' }],
      max_output_tokens: 16000,
    });

    trackUsage(WRITING_MODEL, writeResponse.usage);

    // Count any additional web searches during writing
    const writeCalls = writeResponse.output
      ? writeResponse.output.filter(item => item.type === 'web_search_call').length
      : 0;
    totalWebSearchCalls += writeCalls;
    totalCost += writeCalls * WEB_SEARCH_COST_PER_CALL;

    report = extractText(writeResponse);
    console.log(`[Agent] Report written: ${report.split(/\s+/).length} words`);

  } catch (err) {
    console.error(`[Agent] Writing failed: ${err.message}`);
    throw err;
  }

  // ========== PHASE 4: Self-Review ==========
  console.log(`[Agent] === PHASE 4: Self-review ===`);

  if (!hasTime(30_000)) {
    console.log(`[Agent] Skipping self-review (time pressure)`);
  } else {
    try {
      const reviewResponse = await client.responses.create({
        model: WRITING_MODEL,
        instructions: `You are a strict editorial reviewer. Your job is to find and fix weaknesses.

Check for:
1. UNSOURCED CLAIMS: Any factual statement without a [Source](URL) link -- add the source or flag as "estimate"
2. VAGUE LANGUAGE: Replace "several", "various", "significant" with specific numbers
3. THIN SECTIONS: Any section under 400 words needs expansion with specific data
4. REPETITION: Remove redundant points across sections
5. MISSING CONTRADICTIONS: If only one side of a debatable claim is presented, add the counterargument
6. BROKEN LOGIC: If a conclusion doesn't follow from the evidence, fix it
7. GENERIC RECOMMENDATIONS: Replace "monitor developments" with specific actions, timelines, triggers

Fix ALL issues inline. Return the complete improved report.
Do NOT add commentary -- just return the fixed report.
Do NOT use emojis.`,
        input: [{ role: 'user', content: `Review and improve this research report. Fix all issues inline. Return the complete improved report:\n\n${report}` }],
        tools: [{ type: 'web_search_preview' }],
        max_output_tokens: 16000,
      });

      trackUsage(WRITING_MODEL, reviewResponse.usage);

      const reviewCalls = reviewResponse.output
        ? reviewResponse.output.filter(item => item.type === 'web_search_call').length
        : 0;
      totalWebSearchCalls += reviewCalls;
      totalCost += reviewCalls * WEB_SEARCH_COST_PER_CALL;

      const reviewedReport = extractText(reviewResponse);
      if (reviewedReport && reviewedReport.length > report.length * 0.5) {
        report = reviewedReport;
        console.log(`[Agent] Self-review complete: ${report.split(/\s+/).length} words`);
      } else {
        console.log(`[Agent] Self-review returned insufficient content, keeping original`);
      }

    } catch (err) {
      console.log(`[Agent] Self-review failed: ${err.message}, keeping original report`);
    }
  }

  // ========== Final Cost Summary ==========
  const webSearchCost = totalWebSearchCalls * WEB_SEARCH_COST_PER_CALL;
  const reportWords = report.split(/\s+/).length;
  const elapsedMin = (elapsed() / 60000).toFixed(1);

  console.log(`[Agent] === COMPLETE ===`);
  console.log(`[Agent] ${reportWords} words | ${totalSearchRounds} rounds | ${totalWebSearchCalls} web calls ($${webSearchCost.toFixed(2)}) | ${totalDataPoints} data points | ${totalDomains} domains | $${totalCost.toFixed(2)} total | ${elapsedMin}min`);

  // Update cost tracker
  costTracker.model = RESEARCH_MODEL;
  costTracker.writing_model = WRITING_MODEL;
  costTracker.tokens_in = totalTokensIn;
  costTracker.tokens_out = totalTokensOut;
  costTracker.cost_usd = totalCost;
  costTracker.web_search_calls = totalWebSearchCalls;
  costTracker.web_search_cost_usd = webSearchCost;
  costTracker.search_rounds = totalSearchRounds;
  costTracker.data_points = totalDataPoints;
  costTracker.domains = totalDomains;
  costTracker.time_ms = elapsed();

  return report;
}


// ========== Prompt Builders ==========

function buildSearchPrompt(question, langName, queries, round, state) {
  let prompt = `RESEARCH QUESTION: "${question}"
LANGUAGE: ${langName}
ROUND: ${round}

`;

  if (round > 1 && state.roundSummaries.length > 0) {
    prompt += `ACCUMULATED KNOWLEDGE (${state.dataPoints.length} data points from ${state.roundSummaries.length} previous rounds):

`;
    // Include summaries from all previous rounds (truncated if too long)
    const previousKnowledge = state.roundSummaries
      .map(r => `--- Round ${r.round} (${r.webCalls} searches) ---\n${r.findings}`)
      .join('\n\n');

    // Keep the most recent rounds in full, truncate older ones
    if (previousKnowledge.length > 80000) {
      const recentRounds = state.roundSummaries.slice(-3);
      const olderRounds = state.roundSummaries.slice(0, -3);
      const olderSummary = olderRounds.map(r =>
        `Round ${r.round}: ${r.webCalls} searches, ${r.words} words of findings`
      ).join('\n');
      prompt += `OLDER ROUNDS (summarized):\n${olderSummary}\n\nRECENT ROUNDS (full):\n`;
      prompt += recentRounds.map(r => `--- Round ${r.round} ---\n${r.findings}`).join('\n\n');
    } else {
      prompt += previousKnowledge;
    }

    prompt += '\n\n';
  }

  prompt += `QUERIES TO EXECUTE NOW (search ALL of these):
${queries.map((q, i) => `${i + 1}. ${q}`).join('\n')}

For each search result, extract:
- SPECIFIC data points (numbers, dates, percentages, amounts, named entities)
- Source URL
- Confidence grade: [A] institutional/primary, [B] quality journalism/research, [C] blog/opinion

After all searches, list:
1. ALL specific data points found with sources
2. Any contradictions between sources
3. Key takeaways from this round`;

  return prompt;
}


function buildAnalyzePrompt(question, langName, round, state, checklist) {
  const checklistStatus = checklist.map(c =>
    `  [${c.status === 'done' ? 'x' : c.status === 'partial' ? '~' : ' '}] #${c.id}: ${c.item}`
  ).join('\n');

  const recentFindings = state.roundSummaries.slice(-2).map(r =>
    `Round ${r.round}: ${r.findings.slice(0, 8000)}`
  ).join('\n\n');

  const dataPointsSummary = state.dataPoints.length > 50
    ? `${state.dataPoints.length} total data points. Latest 20:\n` +
      state.dataPoints.slice(-20).map(dp =>
        `- ${dp.fact} [${dp.confidence}] (${dp.source}, round ${dp.round})`
      ).join('\n')
    : state.dataPoints.map(dp =>
        `- ${dp.fact} [${dp.confidence}] (${dp.source}, round ${dp.round})`
      ).join('\n');

  const mandatoryStatus = MANDATORY_SOURCES.map(s =>
    `  ${state.mandatorySourcesCovered.has(s.name) ? '[x]' : '[ ]'} ${s.name}`
  ).join('\n');

  return `RESEARCH QUESTION: "${question}"
LANGUAGE: ${langName}
ROUND: ${round} of max ${state.roundSummaries.length + 5}

CHECKLIST STATUS:
${checklistStatus}

MANDATORY SOURCES:
${mandatoryStatus}

DATA POINTS COLLECTED (${state.dataPoints.length} total):
${dataPointsSummary}

QUERIES ALREADY RUN (${state.sourcesSearched.length}):
${state.sourcesSearched.slice(-30).map(q => `  - ${q}`).join('\n')}

DOMAINS COVERED (${state.domainsHit.size}):
${[...state.domainsHit].join(', ')}

${state.contradictions.length > 0 ? `CONTRADICTIONS FOUND:\n${state.contradictions.map(c => `  - ${c.claim1} vs ${c.claim2}`).join('\n')}\n` : ''}

RECENT FINDINGS:
${recentFindings}

Analyze findings and decide next steps. Generate NEW queries that target GAPS.
Do NOT repeat queries already run.
Include site:-targeted queries for mandatory sources not yet covered.`;
}


function buildWritePrompt(question, langName, state, checklist, templateContent) {
  // Build the knowledge base from all rounds
  const allFindings = state.roundSummaries
    .map(r => r.findings)
    .join('\n\n---\n\n');

  // Truncate if too long but keep structure
  const findingsForWrite = allFindings.length > 120000
    ? allFindings.slice(0, 120000) + '\n\n[... truncated for length ...]'
    : allFindings;

  const dataPointsList = state.dataPoints
    .map(dp => `- ${dp.fact} [${dp.confidence}] [${dp.source}](${dp.url || 'no URL'})`)
    .join('\n');

  const checklistSummary = checklist.map(c =>
    `[${c.status === 'done' ? 'x' : c.status === 'partial' ? '~' : ' '}] ${c.item}`
  ).join('\n');

  let prompt = `WRITE A COMPREHENSIVE RESEARCH REPORT

QUESTION: ${question}
LANGUAGE: ${langName}

RESEARCH SUMMARY:
- ${state.roundSummaries.length} rounds of research conducted
- ${state.dataPoints.length} specific data points collected
- ${state.domainsHit.size} unique source domains
- ${state.mandatorySourcesCovered.size}/${MANDATORY_SOURCES.length} mandatory sources covered
- ${state.contradictions.length} contradictions identified

CHECKLIST COVERAGE:
${checklistSummary}

EXTRACTED DATA POINTS (${state.dataPoints.length}):
${dataPointsList.slice(0, 30000)}

${state.contradictions.length > 0 ? `CONTRADICTIONS TO ADDRESS:\n${state.contradictions.map(c =>
  `- "${c.claim1}" (${c.source1}) vs "${c.claim2}" (${c.source2})`
).join('\n')}\n\n` : ''}

RAW RESEARCH FINDINGS:
${findingsForWrite}
`;

  if (templateContent) {
    prompt += `\nREPORT TEMPLATE/STRUCTURE GUIDE:\n${templateContent.slice(0, 5000)}\n`;
  }

  prompt += `
CRITICAL RULES:
1. Every claim MUST have a source URL in [Name](URL) format
2. Use SPECIFIC data: exact numbers, dates, named entities. Never "several" or "various".
3. Address contradictions explicitly -- present both sides with sources
4. Use Sherman Kent probability language for uncertainty
5. Include at least one table per major section
6. Minimum 8 substantive sections
7. Start with Methodology Note (date, sources, confidence levels, limitations)
8. End with categorized Source List
9. Do NOT use emojis
10. If you need additional data to make a section strong, search the web for it`;

  return prompt;
}


// ========== Utility Functions ==========

function extractText(response) {
  let text = '';
  if (response.output) {
    for (const block of response.output) {
      if (block.type === 'message' && block.content) {
        for (const c of block.content) {
          if (c.type === 'output_text') text += c.text + '\n';
        }
      }
    }
  }
  // Fallback for different response shapes
  if (!text && response.choices) {
    text = response.choices[0]?.message?.content || '';
  }
  return text.trim();
}


function buildFallbackPlan(question) {
  const keywords = question.split(/\s+/).filter(w => w.length > 3).slice(0, 5);
  const topicStr = keywords.join(' ');
  return {
    classification: 'deep-research',
    topic_keywords: keywords,
    checklist: [
      { id: 1, item: 'Current state and latest developments', priority: 'must', status: 'pending' },
      { id: 2, item: 'Key players and stakeholders', priority: 'must', status: 'pending' },
      { id: 3, item: 'Quantitative data and metrics', priority: 'must', status: 'pending' },
      { id: 4, item: 'Recent policy or regulatory changes', priority: 'should', status: 'pending' },
      { id: 5, item: 'Market or financial data', priority: 'should', status: 'pending' },
      { id: 6, item: 'Expert opinions and institutional views', priority: 'should', status: 'pending' },
      { id: 7, item: 'Prediction market odds', priority: 'should', status: 'pending' },
      { id: 8, item: 'Counterarguments and risks', priority: 'must', status: 'pending' },
      { id: 9, item: 'Historical context and trends', priority: 'should', status: 'pending' },
      { id: 10, item: 'Sentiment and public discourse', priority: 'nice', status: 'pending' },
    ],
    data_sources_needed: ['news', 'financial_data', 'institutional', 'sentiment', 'prediction_markets'],
    initial_queries: [
      `${topicStr} latest news 2026`,
      `${topicStr} analysis report`,
      `${topicStr} data statistics`,
      `site:reuters.com ${topicStr}`,
      `site:polymarket.com ${topicStr}`,
    ],
    mandatory_source_queries: [
      `site:polymarket.com ${topicStr}`,
      `site:reddit.com ${topicStr} discussion`,
    ],
  };
}


function extractFallbackAnalysis(text) {
  // Try to extract queries from unstructured text
  const queryPatterns = [
    /(?:next|new|additional)\s*(?:searches|queries)[:\s]*([\s\S]*?)(?:ready|$)/i,
    /\d+\.\s+"([^"]+)"/g,
    /[-*]\s+(.+site:\S+.+)/g,
  ];

  const queries = [];
  for (const pattern of queryPatterns) {
    const matches = text.match(pattern);
    if (matches) {
      for (const m of matches) {
        const cleaned = m.replace(/^[-*\d.\s]+/, '').replace(/^"|"$/g, '').trim();
        if (cleaned.length > 10 && cleaned.length < 200) {
          queries.push(cleaned);
        }
      }
    }
  }

  return {
    new_data_points: [],
    checklist_updates: [],
    contradictions: [],
    gaps: [],
    next_queries: queries.slice(0, 5),
    mandatory_sources_missing: [],
    ready_to_write: queries.length === 0,
    reasoning: 'Fallback analysis from unstructured text',
  };
}
