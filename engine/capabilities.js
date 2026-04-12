// Aurum Compact -- https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 -- see LICENSE

// Capability definition for Aurum Compact.
// OpenAI only. Models are fixed: gpt-5.4-mini (research) + gpt-5.4 (writing/director).

export const FREE = {
  director: true,
  improver: true,
  fullBrief: false,          // bottom line only
  deepMode: false,
  maxDepth: 'standard',
  maxIterations: 2,
  templates: ['scenario-analysis', 'market-research', 'competitor-audit'],
  outputFormats: ['markdown'],
  crashRecovery: false,
  maxConcurrent: 1,
};

export const PRO = {
  director: true,
  improver: true,
  fullBrief: true,
  deepMode: true,
  maxDepth: 'deep',
  maxIterations: 5,
  templates: 'all',
  outputFormats: ['markdown', 'json'],
  crashRecovery: true,
  maxConcurrent: 1,
};

export const ENTERPRISE = { ...PRO };

export function resolveCapabilities(tier) {
  const tiers = { free: FREE, pro: PRO, enterprise: ENTERPRISE };
  const caps = tiers[tier];
  if (!caps) {
    throw new Error(`Unknown tier: "${tier}". Expected one of: free, pro, enterprise`);
  }
  return caps;
}

export function checkCapability(capabilities, feature, value) {
  if (!(feature in capabilities)) {
    return { allowed: false, reason: `Unknown capability: "${feature}"` };
  }

  if (feature === 'templates') {
    if (capabilities.templates === 'all') {
      return { allowed: true, reason: 'All templates available' };
    }
    const allowed = capabilities.templates.includes(value);
    return {
      allowed,
      reason: allowed
        ? `Template "${value}" is available`
        : `Template "${value}" requires Pro. Available: ${capabilities.templates.join(', ')}`,
    };
  }

  const val = capabilities[feature];
  return {
    allowed: Boolean(val),
    reason: val ? `${feature} is enabled` : `${feature} requires Pro`,
  };
}
