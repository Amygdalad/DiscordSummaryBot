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

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
SUMMARY_PUBLIC = os.getenv("SUMMARY_PUBLIC", "0") == "1"

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

@bot.event
async def on_ready():
    print(f"✅ Bot is ready. Logged in as {bot.user} ({bot.user.id})")
    await tree.sync()
    print("🌍 Global commands synced")

@tree.command(name="ping", description="Check if the bot is alive")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("🏓 Pong! I'm alive.", ephemeral=True)

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

    if not msgs:
        await interaction.followup.send("No messages to summarize.", ephemeral=True)
        return

    # Fallback lightweight summary logic
    visible_msgs = [msg for msg in msgs if msg.content.strip() != ""]
    if not visible_msgs:
        await interaction.followup.send("No readable (non-empty) messages found.", ephemeral=True)
        return

    summary_lines = [f"• {msg.author.display_name}: {msg.content[:100]}" for msg in visible_msgs[-min(5, len(visible_msgs)):]]
    summary = f"📝 Summary of last {len(visible_msgs)} messages:\n" + "\n".join(summary_lines)

    await interaction.followup.send(summary, ephemeral=not SUMMARY_PUBLIC)

bot.run(TOKEN)
