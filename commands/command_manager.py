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
        LIMIT = 100
        await interaction.response.defer()
        msg = await interaction.followup.send(content="Getting all old messages, this might take a while...", wait=True)
        afterdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days)
        histogram = np.zeros(days)
        oldestmessagedate = datetime.datetime.now(datetime.timezone.utc)
        while oldestmessagedate > afterdate:
            print(f"Getting {LIMIT} messages for stats")
            messages = [message async for message in interaction.channel.history(after=afterdate, before=oldestmessagedate, oldest_first=False, limit=LIMIT)]
            if (len(messages) == 0):
                break
            for message in messages:
                oldestmessagedate = message.created_at
                if message.created_at < afterdate:
                  break
                if (message.author.id == CommandManager().discordClient.user.id and message.embeds):
                    histogram[(datetime.datetime.now(datetime.timezone.utc) - message.created_at).days] += 1
        dates = [datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(day) for day in range(days)]
        x = np.asarray(dates, dtype='datetime64[s]')
        plt.plot(x, histogram, linewidth=0.7)
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.gca().set_ylim(ymin=0)
        plt.tight_layout()
        plt.title(label=f"OSINT activity in the last {days} days")
        try:
            plt.savefig('media/fig.png', bbox_inches='tight')
            picture = discord.File('media/fig.png')
            await msg.edit(attachments=[picture], content="")
        finally:
             os.remove('media/fig.png')


