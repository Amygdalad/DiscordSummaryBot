"""
Discord Chat Summary Bot (Free / Self-Hosted)
=============================================

A lightweight Discord bot that summarizes recent channel or thread conversations on-demand using a *local* (free) open-source language model (default: `google/flan-t5-small`) or a heuristic fallback summarizer when model inference isn't available.

Why this design?
- **Free**: Runs locally; no paid API keys required. (Optional: point to any HF Transformers text2text model.)
- **Mobile-friendly**: Uses **slash commands** so you can trigger summaries from the Discord mobile app.
- **Scoped Summaries**: Summarize last *N* messages *or* messages from a time window (e.g., last 24h).
- **Chunked Compression Pipeline**: Long chats are broken into chunks → chunk summaries → meta-summary.
- **Heuristic Fallback**: If the model can't load (low RAM host) we produce an extractive bullet summary grouped by speaker & keywords.

---
Quick Start
-----------
1. **Python 3.10+** recommended.
2. Create & invite a Discord bot (https://discord.com/developers/applications) → Bot → Copy Token.
3. Create a **.env** file alongside this script:

   ```bash
   DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE
   SUM_MODEL=google/flan-t5-small        # optional override; any HF text2text model
   DEVICE=cpu                            # or cuda
   GUILD_IDS=123456789012345678,987654321098765432   # optional: speed slash sync to these guilds only
   MAX_FETCH=500                         # optional: hard cap on messages pulled
   ```

4. Install deps (minimal):
   ```bash
   pip install -U discord.py python-dotenv transformers accelerate sentencepiece
   ```
   *Note:* Some models require `torch`, `safetensors`, etc. Install as needed. For CPU-only tiny model:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

5. Run:
   ```bash
   python discord_summary_bot.py
   ```

6. In Discord, type `/summarize` or `/summarize timeframe:24h`.

---
Slash Commands Overview
-----------------------
- `/summarize` – Summarize recent messages in the current channel.
  - **limit** (int, optional, default=100): number of most recent messages to include.
  - **timeframe** (str, optional): e.g., `24h`, `7d`, `2h`, `30m`. Overrides *limit* if both given.
  - **include_bots** (bool, default=False): include bot/system messages.
- `/autosummary enable:true interval:24h` – Enable periodic summaries posted to the channel (requires Manage Messages permission). Stored in a JSON config file.
- `/autosummary disable` – Turn off.

---
Resource Notes
--------------
`flan-t5-small` will run CPU-only on modest hardware but is not state-of-the-art. Swap in a better model if you have GPU. For very long channels or low-RAM hosts, rely on fallback summary.

---
License: MIT
"""

import asyncio
import json
import os
import re
import textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

import discord
from discord import app_commands, Interaction
from discord.ext import commands, tasks

from dotenv import load_dotenv

# Attempt optional ML imports -------------------------------------------------
HF_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HF_AVAILABLE = True
except Exception:  # pragma: no cover - safe fallback
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config Helpers
# ---------------------------------------------------------------------------
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN", "")
MODEL_NAME = os.getenv("SUM_MODEL", "google/flan-t5-small")
DEVICE_STR = os.getenv("DEVICE", "cpu")
MAX_FETCH = int(os.getenv("MAX_FETCH", "1000"))  # hard safety cap on history pulls

# Parse optional guild list (speeds command sync when developing)
GUILD_IDS = []
if os.getenv("GUILD_IDS"):
    GUILD_IDS = [int(x.strip()) for x in os.getenv("GUILD_IDS").split(",") if x.strip().isdigit()]

# Auto-summary config persistence file
AUTOSUM_CFG_FILE = "autosummary_config.json"

# Discord Intents ------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True  # Required to read messages for summarization

bot = commands.Bot(command_prefix="!", intents=intents)

# Global summarizer instance (lazy init)
_SUMMARIZER = None

# ---------------------------------------------------------------------------
# Utility: parse timeframe strings like "24h", "7d", "30m"
# ---------------------------------------------------------------------------
_TIMEFRAME_RE = re.compile(r"^(\d+)([smhdw])$", re.IGNORECASE)

def parse_timeframe(tf: str) -> Optional[timedelta]:
    m = _TIMEFRAME_RE.match(tf.strip())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    return None

# ---------------------------------------------------------------------------
# Message Collection
# ---------------------------------------------------------------------------
async def fetch_messages(
    channel: discord.TextChannel,
    limit: Optional[int] = 100,
    after: Optional[datetime] = None,
    include_bots: bool = False,
) -> List[discord.Message]:
    """Pull recent messages subject to limit/after.

    Discord returns newest → oldest. We reverse before returning.
    """
    if limit is not None:
        limit = min(limit, MAX_FETCH)

    messages: List[discord.Message] = []
    async for msg in channel.history(limit=limit, after=after, oldest_first=False):
        if not include_bots and (msg.author.bot or msg.webhook_id is not None):
            continue
        messages.append(msg)
    messages.reverse()  # oldest first for chronological summarization
    return messages

# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
MENTION_RE = re.compile(r"<@!?(\d+)>")
CODEBLOCK_RE = re.compile(r"```[\s\S]*?```")
INLINE_CODE_RE = re.compile(r"`[^`]+`")
URL_RE = re.compile(r"https?://\S+")


def clean_content(content: str) -> str:
    # Remove code blocks (we'll handle separately maybe later)
    content = CODEBLOCK_RE.sub("[code omitted]", content)
    # Inline code
    content = INLINE_CODE_RE.sub("[inline code]", content)
    # Mentions
    content = MENTION_RE.sub("@user", content)
    # URLs
    content = URL_RE.sub("[link]", content)
    # Collapse whitespace
    content = re.sub(r"\s+", " ", content).strip()
    return content


def serialize_messages(messages: List[discord.Message], max_chars: int = 4000) -> str:
    """Turn a list of messages into a dialogue transcript string.

    We intentionally keep speaker labels short to conserve tokens.
    """
    rows = []
    for m in messages:
        name = m.author.display_name or m.author.name
        ts = m.created_at.strftime("%Y-%m-%d %H:%M")
        body = clean_content(m.content)
        if not body and m.attachments:
            body = "[attachment: {}]".format(", ".join(a.filename for a in m.attachments))
        if not body:
            continue
        rows.append(f"{ts} | {name}: {body}")
    text = "\n".join(rows)
    if len(text) <= max_chars:
        return text
    # Truncate from the front if too long (oldest first already)
    return text[-max_chars:]

# ---------------------------------------------------------------------------
# ML Summarizer Wrapper
# ---------------------------------------------------------------------------
class MLDialogueSummarizer:
    """Chunked summarization pipeline using a seq2seq model."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE_STR):
        self.model_name = model_name
        self.device = 0 if device.startswith("cuda") else -1
        self.pipe = None
        self.max_chunk_chars = 3000  # safe chunk size for small models
        self.chunk_summary_tokens = 128
        self.final_summary_tokens = 256
        self._init_pipe()

    def _init_pipe(self):
        if not HF_AVAILABLE:
            return
        try:
            # Use generic text2text pipeline (works for flan-t5, bart, pegasus, etc.)
            self.pipe = pipeline("text2text-generation", model=self.model_name, device=self.device)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Could not load HF model {self.model_name}: {e}")
            self.pipe = None

    def _model_summary(self, prompt: str, max_new_tokens: int = 128) -> str:
        if not self.pipe:
            return ""
        prompt = prompt.strip()
        if not prompt:
            return ""
        # Many instruction-tuned models respond better with explicit instruction
        if "summarize" not in prompt.lower():
            model_in = "Summarize this Discord conversation in a concise bullet format highlighting key topics, decisions, and action items:\n\n" + prompt
        else:
            model_in = prompt
        try:
            out = self.pipe(model_in, max_new_tokens=max_new_tokens, do_sample=False)
            return out[0]["generated_text"].strip()
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Model inference failed: {e}")
            return ""

    def summarize_dialogue(self, messages: List[discord.Message]) -> str:
        if not messages:
            return "(No messages to summarize.)"
        transcript_full = serialize_messages(messages, max_chars=1000000)  # we'll chunk later
        if not self.pipe:
            return heuristic_summary(messages)
        # Chunk if too long
        chunks = []
        cur = []
        cur_len = 0
        for line in transcript_full.splitlines():
            ln = len(line) + 1
            if cur_len + ln > self.max_chunk_chars and cur:
                chunks.append("\n".join(cur))
                cur = [line]
                cur_len = ln
            else:
                cur.append(line)
                cur_len += ln
        if cur:
            chunks.append("\n".join(cur))
        # Summ each chunk
        chunk_summaries = []
        for i, ch in enumerate(chunks, 1):
            chunk_summaries.append(self._model_summary(ch, max_new_tokens=self.chunk_summary_tokens))
        # Final compress
        combined = "\n\n".join(chunk_summaries)
        final = self._model_summary(combined, max_new_tokens=self.final_summary_tokens)
        if not final.strip():
            return heuristic_summary(messages)
        return final.strip()

# ---------------------------------------------------------------------------
# Heuristic fallback summarizer (extractive)
# ---------------------------------------------------------------------------
def heuristic_summary(messages: List[discord.Message], max_lines: int = 15) -> str:
    """Cheap, no-ML summary: speaker counts + top long lines + attachments note."""
    if not messages:
        return "(No messages.)"
    counts: Dict[str, int] = {}
    attachments = 0
    longlines: List[Tuple[int, str, str]] = []  # (len, author, text)
    for m in messages:
        if m.author.bot:
            continue
        name = m.author.display_name or m.author.name
        counts[name] = counts.get(name, 0) + 1
        attachments += len(m.attachments)
        cleaned = clean_content(m.content)
        if cleaned:
            longlines.append((len(cleaned), name, cleaned))
    longlines.sort(reverse=True)  # longest first acts as proxy for substance

    top_speakers = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    speaker_lines = [f"{n}: {c} msg" for n, c in top_speakers[:8]]

    bullet_lines = ["**Conversation Summary (Heuristic)**", "Top participants: " + ", ".join(speaker_lines)]
    if attachments:
        bullet_lines.append(f"Attachments shared: {attachments}")
    bullet_lines.append("Notable messages:")
    for _, name, text in longlines[: max_lines - len(bullet_lines) - 1]:
        clip = textwrap.shorten(text, width=140, placeholder="…")
        bullet_lines.append(f"- {name}: {clip}")
    return "\n".join(bullet_lines)

# ---------------------------------------------------------------------------
# Summarizer Accessor (lazy global)
# ---------------------------------------------------------------------------

def get_summarizer() -> MLDialogueSummarizer:
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = MLDialogueSummarizer()
    return _SUMMARIZER

# ---------------------------------------------------------------------------
# AUTOSUMMARY CONFIG MANAGEMENT
# ---------------------------------------------------------------------------

def load_autosum_cfg() -> Dict[str, Any]:
    if not os.path.exists(AUTOSUM_CFG_FILE):
        return {}
    try:
        with open(AUTOSUM_CFG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_autosum_cfg(cfg: Dict[str, Any]):
    try:
        with open(AUTOSUM_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to save autosummary config: {e}")

# cfg schema: {"channel_id": {"interval_sec": int, "include_bots": bool}}

_AUTOSUM_CFG_MEM = load_autosum_cfg()

# ---------------------------------------------------------------------------
# AUTOSUMMARY TASK LOOP
# ---------------------------------------------------------------------------
@tasks.loop(minutes=5.0)
async def autosummary_poll():
    """Runs every 5 minutes; posts a summary when interval elapsed."""
    if not _AUTOSUM_CFG_MEM:
        return
    now = datetime.now(timezone.utc)
    to_save = False
    for chan_id, rec in list(_AUTOSUM_CFG_MEM.items()):
        last_iso = rec.get("last_run")
        interval_sec = rec.get("interval_sec", 86400)
        include_bots = rec.get("include_bots", False)
        try:
            chan = bot.get_channel(int(chan_id))
            if chan is None:
                continue
            last_dt = datetime.fromisoformat(last_iso) if last_iso else datetime.fromtimestamp(0, tz=timezone.utc)
            if (now - last_dt).total_seconds() < interval_sec:
                continue
            # time to summarize
            after = now - timedelta(seconds=interval_sec)
            msgs = await fetch_messages(chan, limit=None, after=after, include_bots=include_bots)
            summ = get_summarizer().summarize_dialogue(msgs)
            await chan.send(f"Auto-summary for last {interval_sec//3600 if interval_sec>=3600 else interval_sec//60} {'h' if interval_sec>=3600 else 'm'}:\n{summ}")
            rec["last_run"] = now.isoformat()
            to_save = True
        except Exception as e:  # pragma: no cover
            print(f"[WARN] autosummary_poll error for chan {chan_id}: {e}")
    if to_save:
        save_autosum_cfg(_AUTOSUM_CFG_MEM)

# ---------------------------------------------------------------------------
# SLASH COMMANDS
# ---------------------------------------------------------------------------
class SummaryCommands(app_commands.Group):
    def __init__(self):
        super().__init__(name="summarize", description="Summarize recent chat.")

    @app_commands.command(name="here", description="Summarize this channel.")
    @app_commands.describe(limit="Number of recent messages (max cap)")
    @app_commands.describe(timeframe="Time window e.g. 24h, 7d, 30m")
    @app_commands.describe(include_bots="Include bot/system messages")
    async def summarize_here(
        self,
        interaction: Interaction,
        limit: Optional[int] = 100,
        timeframe: Optional[str] = None,
        include_bots: bool = False,
    ):
        await interaction.response.defer(thinking=True, ephemeral=True)
        channel = interaction.channel  # type: ignore
        if not isinstance(channel, discord.TextChannel):
            await interaction.followup.send("This command must be used in a text channel.", ephemeral=True)
            return

        after_dt = None
        if timeframe:
            delta = parse_timeframe(timeframe)
            if delta is None:
                await interaction.followup.send("Invalid timeframe. Use forms like 24h, 7d, 30m, 2h.", ephemeral=True)
                return
            after_dt = datetime.now(timezone.utc) - delta
            limit = None  # override limit when timeframe given

        msgs = await fetch_messages(channel, limit=limit, after=after_dt, include_bots=include_bots)
        summarizer = get_summarizer()
        summary = summarizer.summarize_dialogue(msgs)

        # Discord ephemeral messages limit ~2000 chars; chunk reply if too long
        MAX_LEN = 1900
        if len(summary) <= MAX_LEN:
            await interaction.followup.send(summary, ephemeral=True)
        else:
            # send as file to avoid truncation
            data = summary.encode("utf-8")
            fp = discord.File(fp=discord.BytesIO(data), filename="summary.txt")
            await interaction.followup.send(content="Summary is long; see attached.", file=fp, ephemeral=True)

# ---------------------------------------------------------------------------
# AUTOSUMMARY SLASH COMMANDS
# ---------------------------------------------------------------------------
class AutoSummaryCommands(app_commands.Group):
    def __init__(self):
        super().__init__(name="autosummary", description="Schedule periodic channel summaries.")

    @app_commands.command(name="set", description="Enable or change auto-summary in this channel.")
    @app_commands.describe(interval="e.g. 24h, 2h, 30m")
    @app_commands.describe(include_bots="Include bot/system messages")
    async def autosum_set(
        self,
        interaction: Interaction,
        interval: str,
        include_bots: bool = False,
    ):
        await interaction.response.defer(thinking=True, ephemeral=True)
        channel = interaction.channel  # type: ignore
        if not isinstance(channel, discord.TextChannel):
            await interaction.followup.send("Use in a text channel.", ephemeral=True)
            return

        delta = parse_timeframe(interval)
        if delta is None:
            await interaction.followup.send("Invalid interval. Use 24h, 2h, 30m, etc.", ephemeral=True)
            return
        secs = int(delta.total_seconds())
        _AUTOSUM_CFG_MEM[str(channel.id)] = {
            "interval_sec": secs,
            "include_bots": include_bots,
            "last_run": datetime.now(timezone.utc).isoformat(),  # start clock now
        }
        save_autosum_cfg(_AUTOSUM_CFG_MEM)
        await interaction.followup.send(f"Auto-summary every {interval} enabled in this channel.", ephemeral=True)

    @app_commands.command(name="off", description="Disable auto-summary in this channel.")
    async def autosum_off(self, interaction: Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        channel = interaction.channel  # type: ignore
        if not isinstance(channel, discord.TextChannel):
            await interaction.followup.send("Use in a text channel.", ephemeral=True)
            return
        _AUTOSUM_CFG_MEM.pop(str(channel.id), None)
        save_autosum_cfg(_AUTOSUM_CFG_MEM)
        await interaction.followup.send("Auto-summary disabled in this channel.", ephemeral=True)

# ---------------------------------------------------------------------------
# BOT EVENTS
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")
    # Sync slash commands
    try:
        if GUILD_IDS:
            for gid in GUILD_IDS:
                guild_obj = discord.Object(id=gid)
                await bot.tree.sync(guild=guild_obj)
            print(f"Synced commands to guilds: {GUILD_IDS}")
        else:
            await bot.tree.sync()
            print("Synced global commands (may take up to 1h to appear globally).")
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Slash sync failed: {e}")

    if not autosummary_poll.is_running():
        autosummary_poll.start()

# Register command groups ----------------------------------------------------
summary_group = SummaryCommands()
autosum_group = AutoSummaryCommands()

bot.tree.add_command(summary_group)
bot.tree.add_command(autosum_group)

# ---------------------------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("DISCORD_TOKEN not set in environment or .env file.")
    # Kick off bot loop
    bot.run(TOKEN)
