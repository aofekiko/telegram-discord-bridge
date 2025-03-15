import datetime
import os
import discord
import discord.ext.commands
from discord import app_commands as apc
from core.singleton import SingletonMeta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class CommandManager(metaclass=SingletonMeta):
    comamndTree: apc.CommandTree
    discordClient: discord.Client

    def __init__(self):
         pass

    def __init__(self, discordClient):
        self.discordClient = discordClient
        self.comamndTree = apc.CommandTree(self.discordClient)
        self.comamndTree.add_command(stats)

@apc.command(name="stats", description="Get a graph of Guy's post frequency")
@apc.describe(days="The number of days back to collect data from")
@apc.rename(days='days')
async def stats(interaction: discord.Interaction, days: int):
        messages = [message async for message in interaction.channel.history(after=(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days)))]
        histogram = np.zeros(days)
        for message in messages:
             if (message.author.id == CommandManager().discordClient.user.id and message.embeds):
                histogram[(datetime.datetime.now(datetime.timezone.utc) - message.created_at).days] += 1
        dates = [datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(day) for day in range(days)]
        x = np.asarray(dates, dtype='datetime64[s]')
        plt.plot(x, histogram, linewidth=0.7)
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        try:
            plt.savefig('media/fig.png')
            picture = discord.File('media/fig.png')
            await interaction.response.send_message(file=picture)
        finally:
             os.remove('media/fig.png')



