// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import { matchTemplate } from './template-matcher.js';
import { researchOpenAI } from './researcher.js';
import { researchAgent } from './researcher-agent.js';
import { improveReport } from './improver.js';
import { generateBottomLine } from './bottom-line.js';
import { evaluate } from './director.js';
import { FREE } from './capabilities.js';
import { SystemMonitor } from './system-monitor.js';

/**
 * Main research orchestrator with full cost tracking.
 * Each request gets its own isolated context — no shared state.
 *
 * Pipeline: question -> template match -> research -> director -> (iterate?) -> bottom line -> result
 */
export async function runResearch(question, keys, config = {}) {
  const startTime = Date.now();
  const timeLimitMs = (config.time_limit_minutes || 60) * 60 * 1000;
  const language = config.language || 'en';
  const caps = config.capabilities || FREE;
  const onProgress = config.onProgress || (() => {});

  // Depth gate
  if (config.depth === 'deep' && !caps.deepMode) {
    console.log('[Orchestrator] Deep mode not available in current tier, using standard');
    config.depth = 'standard';
  }
  const depth = config.depth || 'standard';

  // Iteration gate
  const maxIterations = Math.min(config.max_iterations || 2, caps.maxIterations);

  console.log(`[Orchestrator] Capabilities: director=${caps.director}, improver=${caps.improver}, fullBrief=${caps.fullBrief}, depth=${caps.maxDepth}`);

  // Cost tracking
  const costs = {
    template_match: { model: '', tokens_in: 0, tokens_out: 0, cost_usd: 0, time_ms: 0 },
    research_iterations: [],
    reduce: { model: '', tokens_in: 0, tokens_out: 0, cost_usd: 0, time_ms: 0 },
    director_iterations: [],
    brief: { model: '', tokens_in: 0, tokens_out: 0, cost_usd: 0, time_ms: 0 },
    total_cost_usd: 0,
    total_time_ms: 0,
  };

  console.log(`[Orchestrator] Starting research: "${question.slice(0, 80)}..." (lang=${language}, depth=${depth}, max_iter=${maxIterations})`);

  // System resource monitoring (this process only)
  const monitor = new SystemMonitor();
  monitor.start(3000);

  // Catch unhandled rejections within this research run
  const unhandledHandler = (reason) => {
    console.log(`[Orchestrator] Unhandled rejection caught: ${reason?.message || reason}`);
  };
  process.on('unhandledRejection', unhandledHandler);

  let reportMarkdown = '';
  let directorScore = 0;
  let directorFeedback = '';
  let iteration = 0;
  let templateName = 'general';

  try {
    validateKeys(keys);

    // Step 1: Match template
    const t1 = Date.now();
    try {
      templateName = config.template === 'auto' || !config.template
        ? await matchTemplate(question, keys.openai, costs)
        : config.template;
    } catch (err) {
      console.log(`[Orchestrator] Template matching failed: ${err.message}, using 'general'`);
      templateName = 'general';
    }
    costs.template_match.time_ms = Date.now() - t1;

    // Template gate
    if (caps.templates !== 'all' && !caps.templates.includes(templateName)) {
      console.log(`[Orchestrator] Template "${templateName}" not available, falling back to scenario-analysis`);
      templateName = 'scenario-analysis';
    }

    onProgress('template', { status: 'done', template: templateName });
    console.log(`[Orchestrator] Template: ${templateName} (${costs.template_match.time_ms}ms, $${costs.template_match.cost_usd.toFixed(4)})`);

    // Step 2: Research + Director loop
    while (iteration < maxIterations) {
      iteration++;

      // Time limit only gates iterations AFTER the first
      if (iteration > 1 && (Date.now() - startTime) > timeLimitMs) {
        console.log(`[Orchestrator] Time limit reached, skipping iteration ${iteration}`);
        break;
      }

      const iterCosts = { model: '', tokens_in: 0, tokens_out: 0, cost_usd: 0, time_ms: 0, turn_count: 0, search_count: 0 };
      const t2 = Date.now();

      try {
        if (iteration === 1) {
          // ITERATION 1: Research phase
          const researchTimeBudget = Math.max(timeLimitMs - (Date.now() - startTime), 120_000);
          const researchTimeout = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Research timeout exceeded')), researchTimeBudget + 60_000)
          );

          // Agentic researcher is the default. Use pipeline only if explicitly set.
          const usePipeline = config.researcher === 'pipeline';
          const researchFn = usePipeline ? researchOpenAI : researchAgent;

          reportMarkdown = await Promise.race([
            researchFn(question, keys.openai, {
              templateName,
              language,
              depth,
              timeLimitMs: researchTimeBudget,
              costTracker: iterCosts,
              onProgress,
            }),
            researchTimeout,
          ]);
        } else {
          // ITERATION 2+: Improve existing report in-place using director feedback
          console.log(`[Orchestrator] Improving report in-place (iteration ${iteration}, director score: ${directorScore})`);
          onProgress('improve', { status: 'start', iteration });
          reportMarkdown = await improveReport(reportMarkdown, question, directorFeedback, keys.openai, {
            language,
            costTracker: iterCosts,
          });
          onProgress('improve', { status: 'done', iteration });
        }
      } catch (err) {
        console.log(`[Orchestrator] Research iteration ${iteration} failed: ${err.message}`);
        if (reportMarkdown.length > 0) {
          console.log(`[Orchestrator] Using partial report from previous iteration (${reportMarkdown.length} chars)`);
        }
        iterCosts.time_ms = Date.now() - t2;
        costs.research_iterations.push(iterCosts);
        break;
      }

      iterCosts.time_ms = Date.now() - t2;
      costs.research_iterations.push(iterCosts);
      console.log(`[Orchestrator] ${iteration === 1 ? 'Research' : 'Improvement'} iteration ${iteration}: ${reportMarkdown.split(/\s+/).length} words, ${iterCosts.time_ms}ms, $${iterCosts.cost_usd.toFixed(4)}, ${iterCosts.search_count || 0} searches`);

      // Director evaluation
      if (!caps.director) {
        console.log('[Orchestrator] Director not available in current tier, skipping quality review');
        break;
      }

      try {
        onProgress('director', { status: 'start', iteration });
        const dirCosts = { model: '', tokens_in: 0, tokens_out: 0, cost_usd: 0, time_ms: 0 };
        const t3 = Date.now();
        const dirResult = await evaluate(reportMarkdown, question, keys.openai, dirCosts);
        dirCosts.time_ms = Date.now() - t3;
        costs.director_iterations.push(dirCosts);

        directorScore = dirResult.score;
        directorFeedback = dirResult.feedback;
        onProgress('director', { status: 'done', iteration, score: directorScore, action: dirResult.action });
        console.log(`[Orchestrator] Director: ${dirResult.action} (score: ${directorScore}/100, ${dirCosts.time_ms}ms, $${dirCosts.cost_usd.toFixed(4)})`);

        if (dirResult.action === 'GOAL_MET') {
          console.log(`[Orchestrator] Director says GOAL_MET at iteration ${iteration}`);
          break;
        }

        // Improver gate — only attempt improvement if allowed
        if (!caps.improver) {
          console.log('[Orchestrator] Improver not available in current tier, using current report');
          break;
        }
      } catch (err) {
        console.log(`[Orchestrator] Director evaluation failed: ${err.message}, continuing with current report`);
        break;
      }

      // Check if we have time for another iteration
      if ((Date.now() - startTime) > timeLimitMs) {
        console.log(`[Orchestrator] Time limit reached after director, stopping with score ${directorScore}`);
        break;
      }
    }

    console.log(`[Orchestrator] Research complete: ${reportMarkdown.length} chars after ${iteration} iteration(s), director score: ${directorScore}/100`);

    // Step 3: Generate bottom line — ALWAYS runs (Free tier gets bottom line only)
    try {
      onProgress('brief', { status: 'start' });
      const t4 = Date.now();
      var briefResult = await generateBottomLine(reportMarkdown, keys.openai, costs.brief);
      costs.brief.time_ms = Date.now() - t4;
      onProgress('brief', { status: 'done' });
      console.log(`[Orchestrator] Bottom line generated (${costs.brief.time_ms}ms, $${costs.brief.cost_usd.toFixed(4)})`);
    } catch (err) {
      console.log(`[Orchestrator] Bottom line generation failed: ${err.message}`);
      var briefResult = { bottom_line: reportMarkdown.slice(0, 200) + '...', confidence: { level: 'low', percentage: 20, kent_term: 'Remote' } };
    }
  } catch (err) {
    // Top-level catch: return whatever we have
    console.log(`[Orchestrator] FATAL ERROR: ${err.message}`);
    console.log(`[Orchestrator] Returning partial results (${reportMarkdown.length} chars)`);
    if (reportMarkdown.length === 0) {
      reportMarkdown = `# Research Failed\n\nThe research engine encountered an error: ${err.message}\n\nPlease try again.`;
    }
    var briefResult = { bottom_line: 'Research failed. See report.', confidence: { level: 'low', percentage: 10, kent_term: 'Remote' } };
  } finally {
    // Always clean up
    process.removeListener('unhandledRejection', unhandledHandler);
  }

  const totalTime = Date.now() - startTime;
  costs.total_time_ms = totalTime;
  costs.total_cost_usd =
    costs.template_match.cost_usd +
    costs.research_iterations.reduce((s, i) => s + i.cost_usd, 0) +
    (costs.reduce.cost_usd || 0) +
    costs.director_iterations.reduce((s, i) => s + i.cost_usd, 0) +
    costs.brief.cost_usd;

  const elapsedMinutes = totalTime / 60_000;
  console.log(`[Orchestrator] DONE in ${elapsedMinutes.toFixed(1)} minutes | Total cost: $${costs.total_cost_usd.toFixed(4)} | Iterations: ${iteration}`);

  // Count sources
  const sourceMatches = reportMarkdown.match(/\]\(https?:\/\/[^)]+\)/g) || [];
  const uniqueSources = [...new Set(sourceMatches.map(m => m.slice(2, -1).split('#')[0].split('?')[0]))];

  // Stop system monitor
  const systemStats = monitor.stop();
  console.log(`[Orchestrator] System: CPU avg ${systemStats.cpu_process.avg_pct}%, peak ${systemStats.cpu_process.peak_pct}% | RAM peak ${systemStats.memory_process.rss_peak_mb}MB`);

  return {
    markdown: reportMarkdown,
    brief: briefResult,
    system: systemStats,
    metadata: {
      word_count: reportMarkdown.split(/\s+/).filter(Boolean).length,
      source_count: uniqueSources.length,
      generation_minutes: Math.round(elapsedMinutes * 10) / 10,
      template_used: templateName,
      iterations: iteration,
      director_score: directorScore,
      director_feedback: directorFeedback,
      sources: uniqueSources.slice(0, 200),
    },
    costs,
  };
}

function validateKeys(keys) {
  if (!keys.openai) {
    throw new Error('OpenAI API key required. Set OPENAI_API_KEY or add openai_api_key to .aurum.yaml');
  }
  if (!keys.openai.startsWith('sk-')) {
    throw new Error('Invalid OpenAI API key format (expected sk-...)');
  }
}
