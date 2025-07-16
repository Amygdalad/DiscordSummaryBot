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
async def summarize_here(interaction: discord.Interaction, limit: int = 100):
    channel = interaction.channel
    print(f"🔍 Running summarize in channel: {channel.name}, limit={limit}")

    try:
        msgs = await fetch_messages(channel, limit=limit)
    except discord.Forbidden:
        await interaction.response.send_message(
            "❌ I don't have permission to read messages in this channel. Please check my permissions.",
            ephemeral=True
        )
        return

    if not msgs:
        await interaction.response.send_message("No messages to summarize.", ephemeral=True)
        return

    # Placeholder for summary logic
    summary = "This is a summary of the last {} messages.".format(len(msgs))

    if SUMMARY_PUBLIC:
        await interaction.response.send_message(summary)
    else:
        await interaction.response.send_message(summary, ephemeral=True)

bot.run(TOKEN)