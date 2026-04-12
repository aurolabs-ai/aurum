// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import OpenAI from 'openai';

/**
 * Smart Brief Engine.
 * Parses report sections, scores by source density and analytical depth,
 * pre-selects top findings, then calls GPT-4.1-mini to generate structured brief.
 */
export async function generateBrief(reportMd, openaiKey, costTracker = {}) {
  const parsedSections = parseReportSections(reportMd);
  const preSelectedFindings = selectFindings(parsedSections, 3);

  const openai = new OpenAI({ apiKey: openaiKey });

  const completion = await openai.chat.completions.create({
    model: 'gpt-4.1-mini',
    messages: [{ role: 'system', content: buildBriefPrompt(reportMd, preSelectedFindings) }],
    temperature: 0.2,
    max_completion_tokens: 4500,
  });

  const usage = completion.usage || {};
  costTracker.model = 'gpt-4.1-mini';
  costTracker.tokens_in = usage.prompt_tokens || 0;
  costTracker.tokens_out = usage.completion_tokens || 0;
  costTracker.cost_usd = (costTracker.tokens_in * 0.40 / 1_000_000) + (costTracker.tokens_out * 1.60 / 1_000_000);

  let raw = completion.choices[0].message.content.trim();
  if (raw.startsWith('```')) {
    raw = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
  }

  const brief = JSON.parse(raw);

  // Post-processing: verify section_refs actually exist in the report
  const reportHeadings = parsedSections.map(s => s.heading.toLowerCase());
  if (brief.key_findings) {
    for (const finding of brief.key_findings) {
      if (finding.section_ref) {
        const refLower = finding.section_ref.toLowerCase();
        const exists = reportHeadings.some(h => refLower.includes(h) || h.includes(refLower));
        if (!exists) {
          const match = preSelectedFindings.find(f => f.timeHorizon === finding.time_horizon);
          if (match) finding.section_ref = match.heading;
        }
      }
    }
  }

  return brief;
}

// --- Section parser ---

function parseReportSections(reportMd) {
  const lines = reportMd.split('\n');
  const sections = [];
  let currentHeading = null;
  let buffer = [];
  let inCodeBlock = false;

  function flushSection() {
    if (!currentHeading) return;
    const text = buffer.join('\n').trim();
    if (!text) { currentHeading = null; buffer = []; return; }

    const urlMatches = text.match(/\]\((https?:\/\/[^)]+)\)/g) || [];
    const sources = [...new Set(urlMatches.map(m => m.slice(2, -1).split('#')[0].split('?')[0]))];

    const probRegex = /(Almost certain|Probable|Likely|Roughly even|Unlikely|Remote)\s*\((\d+)%?\s*([^)]*)\)/gi;
    const probabilities = [];
    let pm;
    while ((pm = probRegex.exec(text)) !== null) {
      probabilities.push({ term: pm[1], percentage: parseInt(pm[2]), ci: pm[3]?.trim() || '' });
    }

    const tableRows = text.split('\n').filter(l => l.trim().startsWith('|')).length;
    const numericClaims = (text.match(/\$[\d,.]+|\d+(\.\d+)?%|\d+\s*(mb\/d|bbl|billion|million|trillion|MTPA|GW)/gi) || []).length;
    const wordCount = text.split(/\s+/).filter(Boolean).length;

    sections.push({ heading: currentHeading, text, sources, probabilities, wordCount, tableRows, numericClaims });
    currentHeading = null;
    buffer = [];
  }

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      buffer.push(line);
      continue;
    }
    if (!inCodeBlock) {
      const match = line.match(/^(##)\s+(.+)$/);
      if (match) {
        flushSection();
        currentHeading = match[2].trim();
        continue;
      }
    }
    buffer.push(line);
  }
  flushSection();
  return sections;
}

// --- Section scorer ---

function scoreSections(sections) {
  const totalSections = sections.length;
  return sections.map((s, idx) => {
    const sourceDensity = s.wordCount > 0 ? (s.sources.length / s.wordCount) * 1000 : 0;
    const probScore = Math.min(s.probabilities.length * 2, 10);
    const dataScore = Math.min((s.tableRows * 0.5) + (s.numericClaims * 0.3), 10);
    const positionRatio = idx / totalSections;
    const positionPenalty = positionRatio < 0.2 ? 0.4 : 1.0;

    if (s.wordCount < 50) return { ...s, score: 0 };

    const skipHeadings = /(methodology|sources|references|bibliography|appendix|executive summary|source audit|market data|core market|decision.grade|k[aä]llor|sammanfattning)/i;
    if (skipHeadings.test(s.heading)) return { ...s, score: 0 };

    const rawScore = (sourceDensity * 3) + probScore + dataScore + (s.sources.length * 0.1);
    const score = rawScore * positionPenalty;
    return { ...s, score, sourceDensity, probScore, dataScore, positionPenalty };
  });
}

// --- Finding selector ---

function selectFindings(sections, n = 3) {
  const scored = scoreSections(sections);
  const sorted = scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score);
  const selected = sorted.slice(0, n);

  return selected.map(s => {
    const sentences = s.text
      .replace(/\n/g, ' ')
      .split(/(?<=[.!?])\s+/)
      .filter(sent => sent.length > 30);

    let bestSentence = sentences[0] || s.text.slice(0, 500);
    let bestCitationCount = 0;

    for (const sent of sentences) {
      const citationCount = (sent.match(/\]\(https?:\/\//g) || []).length;
      const refCount = (sent.match(/\]\[[^\]]+\]/g) || []).length;
      const totalCitations = citationCount + refCount;
      if (totalCitations > bestCitationCount) {
        bestCitationCount = totalCitations;
        bestSentence = sent;
      }
    }

    let timeHorizon = 'structural';
    const headingAndText = (s.heading + ' ' + bestSentence).toLowerCase();
    if (/acute|immediate|short.term|0-6 month|days|weeks|current|surge|spike/.test(headingAndText)) {
      timeHorizon = 'acute';
    } else if (/secular|long.term|3-10 year|decade|permanent|structural shift|irreversible/.test(headingAndText)) {
      timeHorizon = 'secular';
    }

    return {
      heading: s.heading,
      claim: bestSentence.trim().slice(0, 600),
      sourceCount: s.sources.length,
      sources: s.sources.slice(0, 5),
      probabilities: s.probabilities,
      timeHorizon,
      score: s.score,
    };
  });
}

// --- Brief prompt builder ---

function buildBriefPrompt(reportMd, preSelectedFindings) {
  let findingsBlock = '';
  if (preSelectedFindings && preSelectedFindings.length > 0) {
    findingsBlock = `
PRE-SELECTED KEY FINDINGS (from section-aware scoring -- use these, do NOT replace with executive summary content):

${preSelectedFindings.map((f, i) => `Finding ${i + 1} -- Section "${f.heading}" (${f.sourceCount} sources, time_horizon: ${f.timeHorizon}):
"${f.claim}"
Source URLs: ${f.sources.slice(0, 3).join(', ')}
${f.probabilities.length > 0 ? `Probability claims in section: ${f.probabilities.map(p => `${p.term} (${p.percentage}%${p.ci ? ' ' + p.ci : ''})`).join(', ')}` : ''}`).join('\n\n')}

IMPORTANT: Your key_findings MUST be based on these 3 pre-selected claims. Formulate each into the key_findings JSON structure. Set section_ref to the section heading shown above. Do NOT replace them with executive summary content or top-of-report claims.
`;
  }

  return `You are a senior intelligence analyst. Read the full report below and produce a structured Decision Brief in JSON format.

The brief must follow the BLUF (Bottom Line Up Front) principle. Be direct, actionable, no hedging.

CRITICAL CALIBRATION RULES:
- The confidence level must reflect the REPORT'S OWN assessment, not your confidence in the summary.
- Use Sherman Kent scale: Almost certain (90-99%), Probable (70-89%), Roughly even (40-69%), Unlikely (11-39%), Remote (1-10%)
- Scenario probabilities must sum to 100% and match the report's own probability estimates where available.
- Every confidence estimate MUST include a confidence interval in percentage points.
- Every finding MUST include source URLs from the report and a per-finding counterargument.
- Every finding MUST include an actor_response: how affected parties are likely to adapt/respond.
- Scenarios MUST include revision_trigger.
${findingsBlock}
Return ONLY valid JSON with this exact structure:
{
  "question": "The central question this report addresses (inferred from the report)",
  "bottom_line": "2-3 sentence direct answer. No hedging. Actionable. Max 50 words.",
  "confidence": {
    "level": "high" | "medium" | "low",
    "kent_term": "Almost certain" | "Probable" | "Roughly even" | "Unlikely" | "Remote",
    "percentage": 55,
    "ci": "+-12pp",
    "source_quality": {
      "primary_institutional": 0,
      "high_secondary": 0,
      "market_signals": 0,
      "news_moderate": 0,
      "total": 0
    },
    "basis": "Brief explanation of confidence basis"
  },
  "key_findings": [
    {
      "finding": "One sentence finding with specific numbers",
      "section_ref": "Section heading from the report",
      "confidence": "high" | "medium" | "low",
      "source_count": 5,
      "source_urls": [{"name": "Source Name", "url": "https://..."}],
      "counterargument": "What could make this finding wrong.",
      "actor_response": "How affected parties are adapting.",
      "time_horizon": "acute" | "structural" | "secular"
    }
  ],
  "time_horizons": {
    "acute": "0-6 months: description",
    "structural": "1-3 years: description",
    "secular": "3-10 years: description"
  },
  "counterargument": {
    "summary": "The strongest case against the bottom line.",
    "strength": "Strong" | "Moderate" | "Weak",
    "what_would_make_this_wrong": "Specific observable that would invalidate the bottom line"
  },
  "scenarios": [
    {
      "name": "Scenario name",
      "probability": "Probable (55% +-10pp)",
      "description": "One sentence",
      "trigger": "What triggers this scenario",
      "type": "base" | "optimistic" | "pessimistic" | "wildcard",
      "key_drivers": ["driver1", "driver2"],
      "revision_trigger": "Observable event that would change this probability by >10pp",
      "early_warning": "Specific indicator to monitor",
      "reference_class": "Of N similar events since YYYY, M (X%) resulted in [outcome]"
    }
  ],
  "recommended_actions": [
    {
      "action": "Specific action",
      "owner": "Who should execute",
      "trigger": "What condition triggers this action",
      "timeline": "By when",
      "impact": "Expected quantified benefit"
    }
  ],
  "not_covered": ["Gap 1", "Gap 2", "Gap 3"],
  "metadata": {
    "source_count": 0,
    "date_range": "",
    "methodology": "Sherman Kent/NATO probability standard"
  }
}

Produce exactly 3 key findings (one per time horizon: acute, structural, secular). Each finding MUST reference a different section of the report.
Produce 4 scenarios (base + optimistic + pessimistic + wildcard), and 3 recommended actions. Max 3 items in not_covered.
Extract actual source URLs from the report markdown.

THE REPORT:
${reportMd.slice(0, 120000)}

Return ONLY the JSON. No markdown fences, no commentary.`;
}
