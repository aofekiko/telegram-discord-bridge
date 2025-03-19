import datetime
import os
import dateparser.conf
import dateutil.parser
import discord
import discord.ext.commands
from discord import app_commands as apc
from core.singleton import SingletonMeta
import matplotlib.pyplot as plt
import numpy as np
import dateutil 
from discord.ext import commands

from bridge.config import Config
from bridge.logger import Logger

config = Config.get_instance()
logger = Logger.get_logger(config.application.name)


class CommandManager(metaclass=SingletonMeta):
    comamndTree: apc.CommandTree
    discordClient: discord.Client

    def __init__(self):
         pass

    def __init__(self, discordClient):
        self.discordClient = discordClient
        self.comamndTree = apc.CommandTree(self.discordClient)
        self.comamndTree.add_command(stats)
        self.comamndTree.add_command(statsbyday)
        self.comamndTree.add_command(statsbyhour)

@apc.command(name="stats", description="Get a graph of OSINT frequency")
@apc.describe(days="The number of days back to collect data from")
@apc.rename(window="moving_average_window")
@apc.rename(days='days')
async def stats(interaction: discord.Interaction, days: int=0, before: str="", after: str="", window: int=0):
        LIMIT = 100

        if (days != 0 and (before != "" or after != "")):
            await interaction.response.send_message(content="Both days and dates are set, only once can be picked")
            return
        if (before == "" and after == "" and days == 0):
            await interaction.response.send_message(content="Either days or the 2 date ranges need to be set")
            return
        if ((before == "" and after != "") or (after == "" and before != "")):
            await interaction.response.send_message(content="Both before and after need to be set")
            return
        
        if (before != "" and after != ""):
            p_before = dateutil.parser.parse(before, dayfirst=True)
            p_after = dateutil.parser.parse(after, dayfirst=True)
            if (p_before < p_after):
                await interaction.response.send_message(content="After date is more recent that the before date")
                return
            if (p_after > datetime.datetime.now()):
                await interaction.response.send_message(content="After date is in the future")
                return
        
        if (days == 0):
            days = (p_before - p_after).days
        await interaction.response.defer()
        msg = await interaction.followup.send(content="Getting all old messages, this might take a while...", wait=True)
        if (after != "" and days == 0): 
            afterdate = p_after
        else:
            afterdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days)
        histogram = np.zeros(days)
        if (before != "" and days == 0):
            oldestmessagedate = p_before
        else:
            oldestmessagedate = datetime.datetime.now(datetime.timezone.utc)
        latestupdate = 0
        while oldestmessagedate > afterdate:
            daydiff = (datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days
            if (latestupdate < daydiff - 5):
                await msg.edit(content=f"Getting all old messages, this might take a while... Day: {(datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days}/{days}")
                latestupdate = daydiff
            logger.debug(f"Getting {LIMIT} messages for stats")
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
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))
        ax.plot(x, histogram, linewidth=1.5)
        if (window != 0):
            average_data = []
            for ind in range(len(histogram) - window + 1):
                average_data.append(np.mean(histogram[ind:ind+window]))
            for ind in range(window - 1):
                average_data.append(np.nan)
            ax.plot(x, average_data,linestyle='dashed' ,linewidth=0.6)
        ax.grid(linestyle=':')
        fig.gca().tick_params(axis='x', labelrotation=45)
        fig.gca().set_ylim(ymin=0)
        fig.gca().set_xmargin(0)
        fig.gca().set_ymargin(0.5)
        fig.tight_layout()
        if (before != "" and after != ""):
            ax.set_title(label=f"OSINT activity between {p_after.strftime("%Y-%m-%d")} and {p_before.strftime("%Y-%m-%d")}")
        else:
            ax.set_title(label=f"OSINT activity in the last {days} days")
        try:
            fig.savefig('media/fig.png', bbox_inches='tight')
            picture = discord.File('media/fig.png')
            await msg.edit(attachments=[picture], content="")
        finally:
             os.remove('media/fig.png')

@apc.command(name="statsbyday", description="Get a bar chart of OSINT activity by weekday")
@apc.describe(days="The number of days back to collect data from")
@apc.rename(days='days')
async def statsbyday(interaction: discord.Interaction, days: int=0, before: str="", after: str=""):
        LIMIT = 100

        if (days != 0 and (before != "" or after != "")):
            await interaction.response.send_message(content="Both days and dates are set, only once can be picked")
            return
        if (before == "" and after == "" and days == 0):
            await interaction.response.send_message(content="Either days or the 2 date ranges need to be set")
            return
        if ((before == "" and after != "") or (after == "" and before != "")):
            await interaction.response.send_message(content="Both before and after need to be set")
            return
        
        if (before != "" and after != ""):
            p_before = dateutil.parser.parse(before, dayfirst=True)
            p_after = dateutil.parser.parse(after, dayfirst=True)
            if (p_before < p_after):
                await interaction.response.send_message(content="After date is more recent that the before date")
                return
            if (p_after > datetime.datetime.now()):
                await interaction.response.send_message(content="After date is in the future")
                return
        
        if (days == 0):
            days = (p_before - p_after).days
        await interaction.response.defer()
        msg = await interaction.followup.send(content="Getting all old messages, this might take a while...", wait=True)
        if (after != "" and days == 0): 
            afterdate = p_after
        else:
            afterdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days)
        data = np.zeros(7)
        if (before != "" and days == 0):
            oldestmessagedate = p_before
        else:
            oldestmessagedate = datetime.datetime.now(datetime.timezone.utc)
        latestupdate = 0
        while oldestmessagedate > afterdate:
            daydiff = (datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days
            if (latestupdate < daydiff - 5):
                await msg.edit(content=f"Getting all old messages, this might take a while... Day: {(datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days}/{days}")
                latestupdate = daydiff
            logger.debug(f"Getting {LIMIT} messages for stats")
            messages = [message async for message in interaction.channel.history(after=afterdate, before=oldestmessagedate, oldest_first=False, limit=LIMIT)]
            if (len(messages) == 0):
                break
            for message in messages:
                oldestmessagedate = message.created_at
                if message.created_at < afterdate:
                  break
                if (message.author.id == CommandManager().discordClient.user.id and message.embeds):
                    data[message.created_at.weekday()] += 1
        dayslist = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))
        ax.bar(dayslist, data)
        ax.grid(linestyle=':')
        fig.gca().set_ylim(ymin=0)
        fig.gca().set_xmargin(0)
        fig.gca().set_ymargin(0.5)
        fig.tight_layout()
        if (before != "" and after != ""):
            ax.set_title(label=f"OSINT activity between {p_after.strftime("%Y-%m-%d")} and {p_before.strftime("%Y-%m-%d")}")
        else:
            ax.set_title(label=f"OSINT activity in the last {days} days")
        try:
            fig.savefig('media/fig.png', bbox_inches='tight')
            picture = discord.File('media/fig.png')
            await msg.edit(attachments=[picture], content="")
        finally:
             os.remove('media/fig.png')

@apc.command(name="statsbyhour", description="Get a bar chart of OSINT activity by hour of the day")
@apc.describe(days="The number of days back to collect data from")
@apc.rename(days='days')
async def statsbyhour(interaction: discord.Interaction, days: int=0, before: str="", after: str=""):
        LIMIT = 100

        if (days != 0 and (before != "" or after != "")):
            await interaction.response.send_message(content="Both days and dates are set, only once can be picked")
            return
        if (before == "" and after == "" and days == 0):
            await interaction.response.send_message(content="Either days or the 2 date ranges need to be set")
            return
        if ((before == "" and after != "") or (after == "" and before != "")):
            await interaction.response.send_message(content="Both before and after need to be set")
            return
        
        if (before != "" and after != ""):
            p_before = dateutil.parser.parse(before, dayfirst=True)
            p_after = dateutil.parser.parse(after, dayfirst=True)
            if (p_before < p_after):
                await interaction.response.send_message(content="After date is more recent that the before date")
                return
            if (p_after > datetime.datetime.now()):
                await interaction.response.send_message(content="After date is in the future")
                return
        
        if (days == 0):
            days = (p_before - p_after).days
        await interaction.response.defer()
        msg = await interaction.followup.send(content="Getting all old messages, this might take a while...", wait=True)
        if (after != "" and days == 0): 
            afterdate = p_after
        else:
            afterdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days)
        data = np.zeros(24)
        if (before != "" and days == 0):
            oldestmessagedate = p_before
        else:
            oldestmessagedate = datetime.datetime.now(datetime.timezone.utc)
        latestupdate = 0
        while oldestmessagedate > afterdate:
            daydiff = (datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days
            if (latestupdate < daydiff - 5):
                await msg.edit(content=f"Getting all old messages, this might take a while... Day: {(datetime.datetime.now(datetime.timezone.utc) - oldestmessagedate).days}/{days}")
                latestupdate = daydiff
            logger.debug(f"Getting {LIMIT} messages for stats")
            messages = [message async for message in interaction.channel.history(after=afterdate, before=oldestmessagedate, oldest_first=False, limit=LIMIT)]
            if (len(messages) == 0):
                break
            for message in messages:
                oldestmessagedate = message.created_at
                if message.created_at < afterdate:
                  break
                if (message.author.id == CommandManager().discordClient.user.id and message.embeds):
                    data[message.created_at.hour] += 1
        hours = np.zeros(24)
        for hour in range(len(hours)):
            hours[hour] = hour

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))
        ax.bar(hours, data)
        ax.grid(linestyle=':', axis='x')
        ax.set_xticks(np.arange(len(hours)))
        fig.gca().set_ylim(ymin=0)
        fig.gca().set_xmargin(0)
        fig.gca().set_ymargin(0.5)
        fig.tight_layout()
        if (before != "" and after != ""):
            ax.set_title(label=f"OSINT activity between {p_after.strftime("%Y-%m-%d")} and {p_before.strftime("%Y-%m-%d")}")
        else:
            ax.set_title(label=f"OSINT activity in the last {days} days")
        try:
            fig.savefig('media/fig.png', bbox_inches='tight')
            picture = discord.File('media/fig.png')
            await msg.edit(attachments=[picture], content="")
        finally:
             os.remove('media/fig.png')