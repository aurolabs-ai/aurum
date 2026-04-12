// Aurum Research Engine — https://github.com/aurolabs-ai/aurum
// Licensed under BSL 1.1 — see LICENSE

import os from 'os';

/**
 * System resource monitor -- measures ONLY this Node process.
 * Call start() before research, stop() after.
 */
export class SystemMonitor {
  constructor() {
    this.samples = [];
    this.interval = null;
    this.startTime = null;
    this.startCpu = null;
  }

  start(intervalMs = 2000) {
    this.samples = [];
    this.startTime = Date.now();
    this.startCpu = process.cpuUsage();
    this.sample();
    this.interval = setInterval(() => this.sample(), intervalMs);
  }

  stop() {
    if (this.interval) clearInterval(this.interval);
    this.interval = null;
    return this.summarize();
  }

  sample() {
    const mem = process.memoryUsage();
    const cpu = process.cpuUsage(this.startCpu);
    const elapsedMs = Date.now() - this.startTime;
    // CPU usage as percentage of elapsed wall time (user + system microseconds)
    const cpuPct = elapsedMs > 0 ? ((cpu.user + cpu.system) / 1000 / elapsedMs) * 100 : 0;

    this.samples.push({
      elapsed_sec: Math.round(elapsedMs / 100) / 10,
      cpu_pct: Math.round(cpuPct * 10) / 10,
      rss_mb: Math.round(mem.rss / 1048576),
      heap_used_mb: Math.round(mem.heapUsed / 1048576),
      heap_total_mb: Math.round(mem.heapTotal / 1048576),
      external_mb: Math.round(mem.external / 1048576),
    });
  }

  summarize() {
    if (this.samples.length === 0) return { error: 'no samples' };

    const rss = this.samples.map(s => s.rss_mb);
    const heap = this.samples.map(s => s.heap_used_mb);
    const cpus = this.samples.map(s => s.cpu_pct);
    const last = this.samples[this.samples.length - 1];

    return {
      duration_sec: last.elapsed_sec,
      sample_count: this.samples.length,
      cpu_process: {
        avg_pct: Math.round(cpus.reduce((a, b) => a + b, 0) / cpus.length * 10) / 10,
        peak_pct: Math.round(Math.max(...cpus) * 10) / 10,
      },
      memory_process: {
        rss_start_mb: this.samples[0].rss_mb,
        rss_peak_mb: Math.max(...rss),
        rss_end_mb: last.rss_mb,
        heap_peak_mb: Math.max(...heap),
        heap_end_mb: last.heap_used_mb,
      },
      system: {
        total_ram_gb: Math.round(os.totalmem() / 1073741824 * 10) / 10,
        free_ram_gb: Math.round(os.freemem() / 1073741824 * 10) / 10,
        cpus: os.cpus().length,
      },
    };
  }
}
