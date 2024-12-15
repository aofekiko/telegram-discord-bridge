
from datetime import datetime, timezone
import glob
import os
import uuid

import aiofiles
from telethon import TelegramClient
from bridge.config import Config
from bridge.logger import Logger

config = Config.get_instance()
logger = Logger.get_logger(config.application.name)


class MediaHandler():

    async def download_media(telegram_client: TelegramClient, event ) -> str:
        os.makedirs(config.application.media_store_location, exist_ok=True)
        media_path = await event.message.download_media(os.path.join(config.application.media_store_location, str(uuid.uuid1())))
        return media_path

    async def append_message_to_file(filename, sent_discord_messages) -> None:
        logger.debug("Saving message data to append only file")
        dated_filename = filename + "-" + datetime.now().replace(tzinfo=timezone.utc).strftime('%Y-%m-%d') + ".txt"
        try:
            async with aiofiles.open(dated_filename, "a", encoding="utf-8") as file:
                for message in sent_discord_messages:
                    if message.embeds[0].description:
                        formatted_message = message.created_at.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%Y/%m/%d, %H:%M:%S") + ": " + message.embeds[0].description + "\n"
                        await file.write(formatted_message)

            logger.debug("Message saved successfully.")

        except Exception as ex:  # pylint: disable=broad-except
            logger.error(
                "An error occurred while saving message: %s", ex, exc_info=config.application.debug)

    def clean_old_media(sent_discord_messages) -> None:
        try:
            for message in sent_discord_messages:
                if message.embeds[0].image:
                    filename = message.embeds[0].image.url.split("/")[-1].split("?")[0] 
                    logger.debug("Removing file: %s", filename)
                    os.remove(os.path.join(config.application.media_store_location ,filename))
        except Exception as ex:
            logger.error("Failed deleting old media file! Make sure that the storage growth does not get out of hand!")

