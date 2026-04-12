// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';

const MODEL = 'gpt-4.1';

const PRICING = {
  'gpt-4.1': { in: 2.0, out: 8.0 },
};

/**
 * Improves an existing report in-place based on director feedback.
 * Parse into sections, identify weak ones, improve each separately, reassemble.
 * Does NOT re-research. Does NOT rewrite from scratch.
 */
export async function improveReport(existingReport, question, directorFeedback, openaiKey, options = {}) {
  const { language = 'en', costTracker = {} } = options;

  const client = new OpenAI({ apiKey: openaiKey });
  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCost = 0;
  let searchCount = 0;

  const langInstruction = language === 'en' ? 'English' : language === 'sv' ? 'Swedish' : language;
  const PRICING_LOCAL = { 'gpt-4.1': { in: 2.0, out: 8.0 }, 'gpt-4.1-mini': { in: 0.4, out: 1.6 } };

  function trackUsage(model, usage) {
    if (!usage) return;
    const ti = usage.prompt_tokens || usage.input_tokens || 0;
    const to = usage.completion_tokens || usage.output_tokens || 0;
    const p = PRICING_LOCAL[model] || PRICING_LOCAL['gpt-4.1'];
    totalTokensIn += ti;
    totalTokensOut += to;
    totalCost += (ti * p.in / 1_000_000) + (to * p.out / 1_000_000);
  }

  const originalWords = existingReport.split(/\s+/).length;
  console.log(`[Improver] Starting sectional improvement (${originalWords} words)`);
  console.log(`[Improver] Director feedback: ${directorFeedback.slice(0, 200)}...`);

  // Step 1: Parse report into sections
  const sections = parseIntoSections(existingReport);
  console.log(`[Improver] Parsed ${sections.length} sections`);

  // Step 2: Identify which sections need improvement from director feedback
  const weakSections = identifyWeakSections(sections, directorFeedback);
  console.log(`[Improver] Weak sections to improve: ${weakSections.length}/${sections.length}`);
  weakSections.forEach(s => console.log(`[Improver]   - "${s.heading}" (${s.wordCount} words): ${s.reason}`));

  if (weakSections.length === 0) {
    console.log(`[Improver] No sections identified for improvement, keeping original`);
    costTracker.model = MODEL;
    costTracker.tokens_in = 0;
    costTracker.tokens_out = 0;
    costTracker.cost_usd = 0;
    return existingReport;
  }

  // Step 3: Improve weak sections in PARALLEL with targeted web search
  let improvedCount = 0;
  const IMPROVE_BATCH = 4;

  for (let batchStart = 0; batchStart < weakSections.length; batchStart += IMPROVE_BATCH) {
    const batch = weakSections.slice(batchStart, batchStart + IMPROVE_BATCH);
    console.log(`[Improver] Batch ${Math.floor(batchStart / IMPROVE_BATCH) + 1}: improving ${batch.length} sections in parallel...`);

    const results = await Promise.allSettled(batch.map(async (weak) => {
      const sectionIndex = sections.findIndex(s => s.heading === weak.heading);
      if (sectionIndex === -1) return null;

      const section = sections[sectionIndex];

      // Step 3a: Targeted search for missing data/verification
      let searchFindings = '';
      try {
        const searchResponse = await client.responses.create({
          model: 'gpt-4.1-mini',
          input: [{ role: 'user', content: `Search for current data to verify and expand this research section about "${section.heading}" related to: ${question}\n\nDirector feedback: ${weak.reason}\n\nFind specific statistics, recent data points, and sources that address the feedback.` }],
          tools: [{ type: 'web_search_preview' }],
          max_output_tokens: 2000,
        });
        const searchText = searchResponse.output?.filter(b => b.type === 'message')
          .flatMap(b => b.content?.filter(c => c.type === 'output_text').map(c => c.text) || [])
          .join('\n') || '';
        if (searchText.length > 50) {
          searchFindings = searchText;
          searchCount++;
          trackUsage('gpt-4.1-mini', searchResponse.usage);
        }
      } catch (e) {
        // Search failed — proceed without new data
      }

      // Step 3b: Improve section with existing content + new search data
      const response = await client.chat.completions.create({
        model: MODEL,
        max_completion_tokens: 4000,
        temperature: 0.3,
        messages: [
          {
            role: 'system',
            content: `You are a senior research editor improving ONE section of a research report. Write in ${langInstruction}.

RULES:
- EXPAND this section with more depth, data, and analysis
- VERIFY existing claims — if new search data contradicts them, note the contradiction
- Keep ALL existing content that is accurate — add to it, do not replace it
- The improved section must be LONGER than the original
- Preserve all existing source URLs and add new ones from the search data
- Add specific numbers, names, dates, source URLs where possible
- Output ONLY the improved section content (no heading)`
          },
          {
            role: 'user',
            content: `# Research question
${question}

# Director feedback for this section
${weak.reason}

${searchFindings ? `# NEW SEARCH DATA (use this to verify claims and fill gaps)\n${searchFindings}\n` : ''}
# CURRENT SECTION: "${section.heading}" (${section.wordCount} words)

${section.content}

Improve this section using the director feedback${searchFindings ? ' and new search data' : ''}. Verify claims. Add depth. Output ONLY the improved content.`
          }
        ],
      });

      trackUsage(MODEL, response.usage);

      const improved = response.choices[0]?.message?.content?.trim() || '';
      const improvedWords = improved.split(/\s+/).length;

      return { sectionIndex, heading: section.heading, originalWords: section.wordCount, improved, improvedWords, hadSearch: !!searchFindings };
    }));

    // Apply results
    for (const result of results) {
      if (result.status !== 'fulfilled' || !result.value) continue;
      const r = result.value;
      if (r.improvedWords >= sections[r.sectionIndex].wordCount * 0.8) {
        sections[r.sectionIndex].content = r.improved;
        sections[r.sectionIndex].wordCount = r.improvedWords;
        improvedCount++;
        console.log(`[Improver]   "${r.heading}": ${r.originalWords} -> ${r.improvedWords} words${r.hadSearch ? ' (+search)' : ''}`);
      } else {
        console.log(`[Improver]   "${r.heading}": too short (${r.improvedWords} vs ${r.originalWords}), keeping original`);
      }
    }
  }

  // Step 4: Reassemble report
  let result = '';
  for (const section of sections) {
    if (section.heading) {
      result += section.headingLine + '\n\n';
    }
    result += section.content + '\n\n';
  }

  const resultWords = result.split(/\s+/).length;
  console.log(`[Improver] Done: ${improvedCount} sections improved, ${searchCount} targeted searches, ${originalWords} -> ${resultWords} words, $${totalCost.toFixed(4)}`);

  costTracker.model = MODEL;
  costTracker.tokens_in = totalTokensIn;
  costTracker.tokens_out = totalTokensOut;
  costTracker.search_count = searchCount;
  costTracker.cost_usd = totalCost;

  return result.trim();
}

/**
 * Parse markdown report into sections by ## headings
 */
function parseIntoSections(report) {
  const lines = report.split('\n');
  const sections = [];
  let current = { heading: '', headingLine: '', content: '', wordCount: 0 };

  for (const line of lines) {
    const h2Match = line.match(/^##\s+(.+)$/);
    const h1Match = line.match(/^#\s+(.+)$/);

    if (h2Match || h1Match) {
      // Save previous section
      if (current.content.trim()) {
        current.wordCount = current.content.split(/\s+/).filter(Boolean).length;
        sections.push({ ...current });
      }
      current = {
        heading: (h2Match || h1Match)[1].trim(),
        headingLine: line,
        content: '',
        wordCount: 0,
      };
    } else {
      current.content += line + '\n';
    }
  }
  // Save last section
  if (current.content.trim()) {
    current.wordCount = current.content.split(/\s+/).filter(Boolean).length;
    sections.push(current);
  }

  return sections;
}

/**
 * Match director feedback to specific sections
 */
function identifyWeakSections(sections, feedback) {
  const feedbackLower = feedback.toLowerCase();
  const weak = [];

  for (const section of sections) {
    const headingLower = section.heading.toLowerCase();
    let reason = '';

    // Check if director mentioned this section by name
    if (feedbackLower.includes(headingLower) || feedbackLower.includes(headingLower.replace(/[^a-z0-9]/g, ' '))) {
      // Extract the relevant feedback line
      const lines = feedback.split('\n');
      for (const line of lines) {
        if (line.toLowerCase().includes(headingLower) || line.toLowerCase().includes(headingLower.replace(/[^a-z0-9]/g, ' '))) {
          reason = line.trim();
          break;
        }
      }
      if (!reason) reason = 'Mentioned in director feedback';
      weak.push({ heading: section.heading, wordCount: section.wordCount, reason });
      continue;
    }

    // Also flag thin sections (<500 words for major topics)
    if (section.wordCount < 500 && section.heading && !section.heading.match(/^(introduction|methodology|table of contents|report structure|source)/i)) {
      reason = `Under-developed section (${section.wordCount} words)`;
      weak.push({ heading: section.heading, wordCount: section.wordCount, reason });
    }
  }

  return weak;
}
