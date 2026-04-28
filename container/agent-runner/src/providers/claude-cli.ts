/**
 * AgentProvider that uses the official Claude Code CLI (`claude -p`) instead
 * of `@anthropic-ai/claude-agent-sdk`.
 *
 * Why this exists:
 * Per Anthropic's Feb 2026 ToS update, subscription tokens
 * (CLAUDE_CODE_OAUTH_TOKEN) are explicitly disallowed in the Agent SDK and
 * any other product that wraps the SDK. The sanctioned headless path for
 * subscription auth is the Claude Code CLI binary in print mode. Users who
 * authenticate against API keys can keep using the existing `claude` provider;
 * users authenticating against a subscription select this provider.
 *
 * Architecture:
 * Requests are forwarded to the `claude-broker` sidecar over a Unix socket
 * (path read from `CLAUDE_BROKER_SOCKET`). The broker owns the actual
 * subprocess spawn, rate-limit detection (5-hour window), per-credential
 * pause-and-queue, and optional API-key fallback. If the broker socket is
 * not configured, the provider falls back to spawning `claude` directly —
 * useful for local tests but does not survive a rate-limit hit gracefully.
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'child_process';
import { connect } from 'net';
import { randomUUID } from 'crypto';

import { registerProvider } from './provider-registry.js';
import type {
  AgentProvider,
  AgentQuery,
  ProviderEvent,
  ProviderOptions,
  QueryInput,
} from './types.js';

function log(msg: string): void {
  console.error(`[claude-cli-provider] ${msg}`);
}

const STALE_SESSION_RE = /no conversation found|ENOENT.*\.jsonl|session.*not found/i;
const RATE_LIMIT_RE = /5-?hour\s+limit|rate\s*limit|\b429\b|temporarily\s+limiting\s+requests/i;

const TOOL_ALLOWLIST = [
  'Bash',
  'Read',
  'Write',
  'Edit',
  'Glob',
  'Grep',
  'WebSearch',
  'WebFetch',
  'Task',
  'TaskOutput',
  'TaskStop',
  'TodoWrite',
  'Skill',
  'NotebookEdit',
  'mcp__nanoclaw__*',
];

const DISALLOWED_TOOLS = [
  'CronCreate',
  'CronDelete',
  'CronList',
  'ScheduleWakeup',
  'AskUserQuestion',
  'EnterPlanMode',
  'ExitPlanMode',
  'EnterWorktree',
  'ExitWorktree',
];

// Stream-json line shapes emitted by `claude -p --output-format stream-json`.
// We deliberately type loosely; new fields appear over Claude Code releases.
type StreamLine = {
  type?: string;
  subtype?: string;
  session_id?: string;
  result?: string;
  message?: {
    content?: Array<{ type: string; text?: string }>;
  };
  [key: string]: unknown;
};

// Broker socket protocol (mirror of claude-broker/src/types.ts in the parent
// repo). Kept inline to avoid coupling the fork to the parent repo's source.
interface BrokerRequest {
  id: string;
  kind: 'invoke';
  role: string;
  credential: string;
  prompt: string;
  cwd: string;
  resume?: string;
  model?: string;
  allowed_tools?: string[];
  disallowed_tools?: string[];
  system_prompt?: string;
}

type BrokerEvent =
  | { id: string; kind: 'init'; session_id: string }
  | { id: string; kind: 'chunk'; line: StreamLine }
  | { id: string; kind: 'done'; tokens?: { input: number; output: number }; text: string | null }
  | { id: string; kind: 'paused'; pause_until: string }
  | { id: string; kind: 'error'; message: string; classification?: string };

class EventQueue {
  private queue: ProviderEvent[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(event: ProviderEvent): void {
    this.queue.push(event);
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *iterate(): AsyncGenerator<ProviderEvent> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

function streamLineToProviderEvents(line: StreamLine): ProviderEvent[] {
  const events: ProviderEvent[] = [{ type: 'activity' }];

  if (line.type === 'system' && line.subtype === 'init' && line.session_id) {
    events.push({ type: 'init', continuation: line.session_id });
  } else if (line.type === 'result') {
    events.push({ type: 'result', text: line.result ?? null });
  } else if (line.type === 'system' && line.subtype === 'rate_limit_event') {
    events.push({
      type: 'error',
      message: 'Rate limit hit',
      retryable: false,
      classification: 'quota',
    });
  } else if (line.type === 'system' && line.subtype === 'compact_boundary') {
    events.push({ type: 'result', text: 'Context compacted.' });
  }
  return events;
}

export class ClaudeCliProvider implements AgentProvider {
  readonly supportsNativeSlashCommands = true;

  private assistantName?: string;
  private brokerSocketPath?: string;
  private credential: string;
  private model: string;
  private env: Record<string, string | undefined>;

  constructor(options: ProviderOptions = {}) {
    this.assistantName = options.assistantName;
    this.env = options.env ?? {};
    this.brokerSocketPath =
      this.env.CLAUDE_BROKER_SOCKET ?? process.env.CLAUDE_BROKER_SOCKET ?? undefined;
    this.credential = (this.env.AGENT_GROUP ?? process.env.AGENT_GROUP ?? 'default').toUpperCase();
    this.model = this.env.CLAUDE_MODEL ?? 'claude-sonnet-4-6';
  }

  isSessionInvalid(err: unknown): boolean {
    const msg = err instanceof Error ? err.message : String(err);
    return STALE_SESSION_RE.test(msg);
  }

  query(input: QueryInput): AgentQuery {
    const events = new EventQueue();
    let aborted = false;

    if (this.brokerSocketPath) {
      void this.runViaBroker(input, events).catch((err) => {
        log(`broker invocation failed: ${err instanceof Error ? err.message : String(err)}`);
        events.push({
          type: 'error',
          message: err instanceof Error ? err.message : String(err),
          retryable: false,
        });
        events.end();
      });
    } else {
      void this.runDirect(input, events).catch((err) => {
        log(`direct invocation failed: ${err instanceof Error ? err.message : String(err)}`);
        events.push({
          type: 'error',
          message: err instanceof Error ? err.message : String(err),
          retryable: false,
        });
        events.end();
      });
    }

    return {
      // Multi-turn push is not yet wired — claude -p in print mode is a
      // single-shot invocation. The poll-loop calls push() for follow-up
      // messages within the same SDK query; in CLI mode each follow-up
      // becomes a fresh `claude --resume <session_id>` invocation. For the
      // MVP we treat push() as a no-op and rely on the poll-loop's
      // continuation flow to thread session_ids across queries.
      push: (msg) => {
        log(`push() called with ${msg.length} chars — ignored in CLI mode`);
      },
      end: () => {
        events.end();
      },
      events: events.iterate(),
      abort: () => {
        aborted = true;
        events.end();
      },
    };

    function _markAborted(): boolean {
      return aborted;
    }
  }

  private async runViaBroker(input: QueryInput, events: EventQueue): Promise<void> {
    if (!this.brokerSocketPath) throw new Error('broker socket not configured');
    const requestId = randomUUID();
    const request: BrokerRequest = {
      id: requestId,
      kind: 'invoke',
      role: this.credential.toLowerCase(),
      credential: this.credential,
      prompt: input.prompt,
      cwd: input.cwd,
      resume: input.continuation,
      model: this.model,
      allowed_tools: TOOL_ALLOWLIST,
      disallowed_tools: DISALLOWED_TOOLS,
      system_prompt: input.systemContext?.instructions,
    };

    await new Promise<void>((resolve, reject) => {
      const sock = connect(this.brokerSocketPath!, () => {
        sock.write(JSON.stringify(request) + '\n');
      });

      let buf = '';
      sock.setEncoding('utf-8');

      sock.on('data', (chunk: string) => {
        buf += chunk;
        let nl: number;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const raw = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          if (!raw.trim()) continue;
          let evt: BrokerEvent;
          try {
            evt = JSON.parse(raw) as BrokerEvent;
          } catch (err) {
            log(`malformed broker frame: ${raw.slice(0, 200)}`);
            continue;
          }
          if (evt.id !== requestId) continue;

          if (evt.kind === 'init') {
            events.push({ type: 'init', continuation: evt.session_id });
          } else if (evt.kind === 'chunk') {
            for (const e of streamLineToProviderEvents(evt.line)) events.push(e);
          } else if (evt.kind === 'done') {
            events.push({ type: 'result', text: evt.text });
            events.end();
            sock.end();
            resolve();
            return;
          } else if (evt.kind === 'paused') {
            events.push({
              type: 'error',
              message: `Rate-limited. Paused until ${evt.pause_until}.`,
              retryable: true,
              classification: 'quota',
            });
            events.end();
            sock.end();
            resolve();
            return;
          } else if (evt.kind === 'error') {
            events.push({
              type: 'error',
              message: evt.message,
              retryable: false,
              classification: evt.classification,
            });
            events.end();
            sock.end();
            resolve();
            return;
          }
        }
      });

      sock.on('error', (err) => reject(err));
      sock.on('close', () => {
        events.end();
        resolve();
      });
    });
  }

  private async runDirect(input: QueryInput, events: EventQueue): Promise<void> {
    const args = [
      '-p',
      '--output-format',
      'stream-json',
      '--input-format',
      'text',
      '--model',
      this.model,
    ];
    if (input.continuation) {
      args.push('--resume', input.continuation);
    }
    if (input.systemContext?.instructions) {
      args.push('--append-system-prompt', input.systemContext.instructions);
    }
    args.push('--allowed-tools', TOOL_ALLOWLIST.join(','));
    args.push('--disallowed-tools', DISALLOWED_TOOLS.join(','));

    const child: ChildProcessWithoutNullStreams = spawn('claude', args, {
      cwd: input.cwd,
      env: { ...process.env, ...this.env } as NodeJS.ProcessEnv,
    });

    child.stdin.write(input.prompt);
    child.stdin.end();

    let stdoutBuf = '';
    let stderrBuf = '';
    child.stdout.setEncoding('utf-8');
    child.stderr.setEncoding('utf-8');

    child.stdout.on('data', (chunk: string) => {
      stdoutBuf += chunk;
      let nl: number;
      while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
        const raw = stdoutBuf.slice(0, nl);
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (!raw.trim()) continue;
        try {
          const line = JSON.parse(raw) as StreamLine;
          for (const e of streamLineToProviderEvents(line)) events.push(e);
        } catch {
          /* skip non-JSON lines */
        }
      }
    });

    child.stderr.on('data', (chunk: string) => {
      stderrBuf += chunk;
    });

    await new Promise<void>((resolve) => {
      child.on('close', (code) => {
        const combined = stdoutBuf + '\n' + stderrBuf;
        if (RATE_LIMIT_RE.test(combined)) {
          events.push({
            type: 'error',
            message: 'Rate-limited by Claude. Inspect output for reset window.',
            retryable: true,
            classification: 'quota',
          });
        } else if (code !== 0) {
          events.push({
            type: 'error',
            message: `claude exited ${code}: ${stderrBuf.slice(0, 500)}`,
            retryable: false,
          });
        }
        events.end();
        resolve();
      });
    });
  }
}

registerProvider('claude-cli', (opts) => new ClaudeCliProvider(opts));
