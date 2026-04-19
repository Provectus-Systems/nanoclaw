/**
 * Step: onecli — Install + configure the OneCLI gateway and CLI.
 *
 * Aggregates what the old /setup + /init-onecli skills ran as loose shell
 * commands. Idempotent: skips install if `onecli` already works, and safely
 * re-applies PATH, api-host, and .env updates.
 *
 * Emits ONECLI_URL so /new-setup SKILL.md can forward it downstream (e.g. as
 * ${ONECLI_URL} in status messages). Polls /health to give downstream steps
 * (auth, service) a ready gateway.
 */
import { execFileSync, execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { log } from '../src/log.js';
import { emitStatus } from './status.js';

const LOCAL_BIN = path.join(os.homedir(), '.local', 'bin');

function childEnv(): NodeJS.ProcessEnv {
  const parts = [LOCAL_BIN];
  if (process.env.PATH) parts.push(process.env.PATH);
  return { ...process.env, PATH: parts.join(path.delimiter) };
}

function onecliVersion(): string | null {
  try {
    return execFileSync('onecli', ['version'], {
      encoding: 'utf-8',
      env: childEnv(),
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return null;
  }
}

function getApiHost(): string | null {
  try {
    const out = execFileSync('onecli', ['config', 'get', 'api-host'], {
      encoding: 'utf-8',
      env: childEnv(),
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    const parsed = JSON.parse(out) as { value?: unknown };
    return typeof parsed.value === 'string' && parsed.value ? parsed.value : null;
  } catch {
    return null;
  }
}

function extractUrlFromOutput(output: string): string | null {
  const match = output.match(/https?:\/\/[\w.\-]+(?::\d+)?/);
  return match ? match[0] : null;
}

function ensureShellProfilePath(): void {
  const home = os.homedir();
  const line = 'export PATH="$HOME/.local/bin:$PATH"';
  for (const profile of [path.join(home, '.bashrc'), path.join(home, '.zshrc')]) {
    try {
      const content = fs.existsSync(profile) ? fs.readFileSync(profile, 'utf-8') : '';
      if (!content.includes('.local/bin')) {
        fs.appendFileSync(profile, `\n${line}\n`);
        log.info('Added ~/.local/bin to PATH in shell profile', { profile });
      }
    } catch (err) {
      log.warn('Could not update shell profile', { profile, err });
    }
  }
}

function writeEnvOnecliUrl(url: string): void {
  const envFile = path.join(process.cwd(), '.env');
  let content = fs.existsSync(envFile) ? fs.readFileSync(envFile, 'utf-8') : '';
  if (/^ONECLI_URL=/m.test(content)) {
    content = content.replace(/^ONECLI_URL=.*$/m, `ONECLI_URL=${url}`);
  } else {
    content = content.trimEnd() + (content ? '\n' : '') + `ONECLI_URL=${url}\n`;
  }
  fs.writeFileSync(envFile, content);
}

function installOnecli(): { stdout: string; ok: boolean } {
  // OneCLI's own install script handles gateway + CLI + PATH.
  // We run the two canonical installers in sequence and capture stdout so
  // we can extract the printed URL as a fallback to `onecli config get`.
  let stdout = '';
  try {
    stdout += execSync('curl -fsSL onecli.sh/install | sh', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    stdout += execSync('curl -fsSL onecli.sh/cli/install | sh', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { stdout, ok: true };
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string };
    log.error('OneCLI install failed', { stderr: e.stderr });
    return { stdout: stdout + (e.stdout ?? '') + (e.stderr ?? ''), ok: false };
  }
}

async function pollHealth(url: string, timeoutMs: number): Promise<boolean> {
  // `/api/health` matches the path probe.mjs uses — keep them aligned.
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/api/health`);
      if (res.ok) return true;
    } catch {
      // not ready yet
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return false;
}

export async function run(_args: string[]): Promise<void> {
  ensureShellProfilePath();

  let installOutput = '';
  let present = !!onecliVersion();
  if (!present) {
    log.info('Installing OneCLI gateway and CLI');
    const res = installOnecli();
    installOutput = res.stdout;
    if (!res.ok) {
      emitStatus('ONECLI', {
        INSTALLED: false,
        STATUS: 'failed',
        ERROR: 'install_failed',
        LOG: 'logs/setup.log',
      });
      process.exit(1);
    }
    present = !!onecliVersion();
    if (!present) {
      emitStatus('ONECLI', {
        INSTALLED: false,
        STATUS: 'failed',
        ERROR: 'onecli_not_on_path_after_install',
        HINT: 'Open a new shell or run `export PATH="$HOME/.local/bin:$PATH"` and retry.',
        LOG: 'logs/setup.log',
      });
      process.exit(1);
    }
  }

  let url = getApiHost();
  if (!url && installOutput) {
    url = extractUrlFromOutput(installOutput);
    if (url) {
      try {
        execFileSync('onecli', ['config', 'set', 'api-host', url], {
          stdio: 'ignore',
          env: childEnv(),
        });
      } catch (err) {
        log.warn('onecli config set api-host failed', { err });
      }
    }
  }

  if (!url) {
    emitStatus('ONECLI', {
      INSTALLED: true,
      STATUS: 'failed',
      ERROR: 'could_not_resolve_api_host',
      HINT: 'Run `onecli config get api-host` to inspect the gateway URL.',
      LOG: 'logs/setup.log',
    });
    process.exit(1);
  }

  writeEnvOnecliUrl(url);
  log.info('Wrote ONECLI_URL to .env', { url });

  const healthy = await pollHealth(url, 15000);

  emitStatus('ONECLI', {
    INSTALLED: true,
    ONECLI_URL: url,
    HEALTHY: healthy,
    // Install succeeded regardless — a failed health poll often just means
    // the endpoint is auth-gated or the gateway hasn't finished warming up.
    // The next step (auth) will surface a genuinely broken gateway via
    // `onecli secrets list`, so don't trigger rescue attempts from here.
    STATUS: 'success',
    ...(healthy
      ? {}
      : {
          HEALTH_HINT:
            'Health poll returned non-ok within 15s — likely auth-gated. Proceed to the auth step; it will surface a real outage.',
        }),
    LOG: 'logs/setup.log',
  });
}
