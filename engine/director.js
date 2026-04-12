// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';

const MODEL = 'gpt-4.1';

const PRICING = {
  'gpt-4.1': { in: 2.0, out: 8.0 },
};

/**
 * Adversarial Director: attacks report quality.
 * NOT a grader. An adversarial attacker that finds weaknesses.
 * Score 85+ = GOAL_MET, anything below = CONTINUE with section-by-section feedback.
 */
export async function evaluate(reportMd, question, openaiKey, costTracker = {}) {
  const startTime = Date.now();
  console.log(`[Director] === ADVERSARIAL EVALUATION ===`);

  const client = new OpenAI({ apiKey: openaiKey });

  const reportWords = reportMd.split(/\s+/).length;
  const urlMatches = reportMd.match(/\]\(https?:\/\/[^)]+\)/g) || [];
  const uniqueUrls = [...new Set(urlMatches.map(m => m.slice(2, -1).split('#')[0].split('?')[0]))];
  const uniqueDomains = [...new Set(uniqueUrls.map(u => { try { return new URL(u).hostname; } catch { return u; } }))];

  // Domain concentration check
  const domainCounts = {};
  for (const d of uniqueUrls.map(u => { try { return new URL(u).hostname; } catch { return u; } })) {
    domainCounts[d] = (domainCounts[d] || 0) + 1;
  }
  const topDomain = Object.entries(domainCounts).sort((a, b) => b[1] - a[1])[0];
  const topDomainPct = topDomain ? Math.round((topDomain[1] / uniqueUrls.length) * 100) : 0;

  // Section analysis
  const sectionHeaders = reportMd.match(/^##\s+.+$/gm) || [];
  const shortSections = [];
  const sectionLines = reportMd.split('\n');
  let currentSection = null;
  let sectionWordCount = 0;

  for (const line of sectionLines) {
    const headerMatch = line.match(/^##\s+(.+)$/);
    if (headerMatch) {
      if (currentSection && sectionWordCount < 500) {
        shortSections.push({ name: currentSection, words: sectionWordCount });
      }
      currentSection = headerMatch[1];
      sectionWordCount = 0;
    } else {
      sectionWordCount += line.split(/\s+/).filter(Boolean).length;
    }
  }
  if (currentSection && sectionWordCount < 500) {
    shortSections.push({ name: currentSection, words: sectionWordCount });
  }

  const preAnalysis = `
PRE-COMPUTED METRICS (verified, use these):
- Report word count: ${reportWords}
- Unique source URLs: ${uniqueUrls.length}
- Unique source domains: ${uniqueDomains.length}
- Top domain: ${topDomain ? `${topDomain[0]} (${topDomainPct}% of sources)` : 'none'}
- Domain concentration: ${topDomainPct > 50 ? 'FLAGGED - over 50% from single domain' : 'acceptable'}
- Sections found: ${sectionHeaders.length}
- Under-developed sections (<500 words): ${shortSections.length > 0 ? shortSections.map(s => `"${s.name}" (${s.words}w)`).join(', ') : 'none'}
`;

  const systemPrompt = `You are an ADVERSARIAL ATTACKER evaluating a research report. Your job is to DESTROY weak reports and only pass genuinely excellent ones.

You are NOT a grader. You are a hostile reviewer. A score of 92 means the report is genuinely excellent — comparable to a professional analyst's output. Most reports should score 50-75.

${preAnalysis}

SCORING BREAKDOWN (100 points total):

1. SOURCES (20 points):
   - 20: 40+ unique URLs from 15+ domains, mix of institutional/industry/news
   - 15: 25-39 URLs from 10+ domains
   - 10: 15-24 URLs from 5+ domains
   - 5: <15 URLs or heavy domain concentration
   - 0: <5 URLs or mostly fabricated

2. DEPTH (20 points):
   - 20: 8000+ words, every section has 500+ words, tables, probability assessments
   - 15: 5000-8000 words, most sections substantive
   - 10: 3000-5000 words, some thin sections
   - 5: <3000 words
   - 0: <1500 words

3. ACTIONABILITY (20 points):
   - 20: Specific recommendations with timelines, named actors, quantified impacts
   - 15: Clear recommendations but some lack specificity
   - 10: Generic recommendations ("monitor developments")
   - 5: No real recommendations
   - 0: No actionable content

4. ADVERSARIAL RIGOR (20 points):
   - 20: Addresses counter-arguments, flags contradictions, uses probability language, identifies gaps
   - 15: Some counter-arguments, probability estimates present
   - 10: Mostly one-sided analysis
   - 5: No counter-arguments or uncertainty acknowledgment
   - 0: Reads like marketing copy

5. INFORMATION DENSITY (20 points):
   - 20: Every paragraph has specific data points (numbers, dates, names, percentages)
   - 15: Most paragraphs data-backed
   - 10: Mix of data and filler
   - 5: Mostly generalities
   - 0: Almost no specific data

THRESHOLD: 85+ = GOAL_MET. Below 85 = CONTINUE.

YOUR OUTPUT MUST BE EXACTLY THIS FORMAT:

Line 1: GOAL_MET or CONTINUE
Line 2: score: XX | sources: XX/20 | depth: XX/20 | actionability: XX/20 | rigor: XX/20 | density: XX/20
Line 3-7: TOP 5 WEAKNESSES (numbered, specific, with section names)
Line 8+: SECTION-BY-SECTION IMPROVEMENT INSTRUCTIONS (only if CONTINUE)

Each weakness must be specific: "Section 'Market Analysis' has 3 claims without source URLs" NOT "add more sources".
Each improvement instruction must name a section and say exactly what to add.`;

  const response = await client.chat.completions.create({
    model: MODEL,
    max_completion_tokens: 2000,
    temperature: 0.1,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `QUESTION: ${question}\n\nREPORT (${reportWords} words, ${uniqueUrls.length} sources):\n\n${reportMd.slice(0, 100000)}` },
    ],
  });

  const usage = response.usage || {};
  const tokIn = usage.prompt_tokens || 0;
  const tokOut = usage.completion_tokens || 0;
  const p = PRICING[MODEL];
  const cost = (tokIn * p.in / 1_000_000) + (tokOut * p.out / 1_000_000);

  costTracker.model = MODEL;
  costTracker.tokens_in = tokIn;
  costTracker.tokens_out = tokOut;
  costTracker.cost_usd = cost;

  const text = response.choices[0]?.message?.content?.trim() || '';
  const lines = text.split('\n').filter(l => l.trim());

  const action = lines[0]?.trim().startsWith('GOAL_MET') ? 'GOAL_MET' : 'CONTINUE';
  const scoreLine = lines[1] || lines[0] || '';
  const scoreMatch = scoreLine.match(/score:\s*(\d+)/i);
  const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

  // Collect all feedback lines after the score line
  const feedbackLines = lines.slice(2);
  const feedback = feedbackLines.join('\n').trim();

  const elapsedMs = Date.now() - startTime;
  console.log(`[Director] ${action} | Score: ${score}/100 | ${elapsedMs}ms | $${cost.toFixed(4)}`);
  console.log(`[Director] Metrics: ${reportWords} words, ${uniqueUrls.length} URLs, ${uniqueDomains.length} domains`);
  if (shortSections.length > 0) {
    console.log(`[Director] Thin sections: ${shortSections.map(s => `${s.name}(${s.words}w)`).join(', ')}`);
  }

  return { action, score, feedback };
}
