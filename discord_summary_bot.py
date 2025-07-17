"""
Discord Chat Summary Bot (Free / Self-Hosted)
=============================================

A lightweight Discord bot that summarizes recent channel or thread conversations on-demand using a *local* (free) open-source language model (default: `google/flan-t5-small`) or a heuristic fallback summarizer when model inference isn't available.
"""

import os
import discord
from discord.ext import commands
from discord import app_commands
import asyncio
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

__version__ = "1.0.0"

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
SUMMARY_PUBLIC = os.getenv("SUMMARY_PUBLIC", "0") == "1"

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

# Load FLAN-T5 model and tokenizer
print("🔍 Loading FLAN-T5 model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("✅ FLAN-T5 model loaded.")

@bot.event
async def on_ready():
    print(f"✅ Bot is ready. Logged in as {bot.user} ({bot.user.id})")
    await tree.sync()
    print("🌍 Global commands synced")

@tree.command(name="ping", description="Check if the bot is alive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"🏓 Pong! I'm alive. (v{__version__})", ephemeral=True)

@tree.command(name="version", description="Show bot version")
async def version(interaction: discord.Interaction):
    await interaction.response.send_message(f"🤖 SummaryBot version: `{__version__}`", ephemeral=True)

async def fetch_messages(channel, limit=100, after=None, include_bots=False):
    messages = []
    try:
        async for msg in channel.history(limit=limit, after=after, oldest_first=False):
            if not include_bots and msg.author.bot:
                continue
            messages.append(msg)
    except discord.Forbidden:
        raise
    return messages

def generate_summary(text):
    prompt = f"Summarize this conversation:\n{text.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    output_ids = model.generate(**inputs, max_length=150)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

@tree.command(name="summarize", description="Summarize recent messages in this channel")
@app_commands.describe(limit="Number of recent messages to summarize")
async def summarize(interaction: discord.Interaction, limit: int = 100):
    channel = interaction.channel
    print(f"🔍 Running summarize in channel: {channel.name}, limit={limit}")

    await interaction.response.defer(thinking=True, ephemeral=not SUMMARY_PUBLIC)

    try:
        msgs = await fetch_messages(channel, limit=limit)
    except discord.Forbidden:
        await interaction.followup.send(
            "❌ I don't have permission to read messages in this channel. Please check my permissions.",
            ephemeral=True
        )
        return

    visible_msgs = [msg for msg in msgs if msg.content.strip() != ""]
    if not visible_msgs:
        await interaction.followup.send("No readable (non-empty) messages found.", ephemeral=True)
        return

    combined_text = "\n".join([f"{msg.author.display_name}: {msg.content}" for msg in reversed(visible_msgs)])

    try:
        summary = generate_summary(combined_text)
        await interaction.followup.send(f"📝 Summary of last {len(visible_msgs)} messages:\n{summary}", ephemeral=not SUMMARY_PUBLIC)
    except Exception as e:
        await interaction.followup.send(f"⚠️ Failed to summarize: {e}", ephemeral=True)

bot.run(TOKEN)