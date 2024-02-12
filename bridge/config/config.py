"""Configuration handler."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import yaml


class AppConfig():  # pylint: disable=too-few-public-methods
    """Application configuration handler."""

    def __init__(self, config_data):
        self.name: str = config_data["name"]
        self.version: str = config_data["version"]
        self.description: str = config_data["description"]
        self.debug: bool = config_data["debug"]
        self.internet_connected: bool = False
        self.healthcheck_interval = config_data["healthcheck_interval"]
        self.recoverer_delay: float = config_data["recoverer_delay"]
        self.history_size_limit: float = config_data["history_size_limit"]


class APIConfig():  # pylint: disable=too-few-public-methods
    """API configuration handler."""

    def __init__(self, config_data):
        self.enabled = config_data["enabled"]
        self.cors_origins: List[str] = config_data["cors_origins"]
        self.telegram_login_enabled: bool = config_data["telegram_login_enabled"]
        self.telegram_auth_file: str = config_data["telegram_auth_file"]
        self.telegram_auth_request_expiration: int = config_data["telegram_auth_request_expiration"]


class LoggerConfig():  # pylint: disable=too-few-public-methods
    """Logger configuration handler."""

    def __init__(self, config_data):
        self.level = config_data["level"]
        self.file_max_bytes = config_data["file_max_bytes"]
        self.file_backup_count = config_data["file_backup_count"]
        self.format = config_data["format"]
        self.date_format = config_data["date_format"]
        self.console = config_data["console"]


class TelegramConfig():  # pylint: disable=too-few-public-methods
    """Telegram configuration handler."""

    def __init__(self, config_data):
        self.is_healthy: bool = False
        self.phone = config_data["phone"]
        self.password: str = config_data["password"]
        self.api_id: int = config_data["api_id"]
        self.api_hash: str = config_data["api_hash"]
        self.log_unhandled_conversations: bool = config_data["log_unhandled_conversations"]


class DiscordConfig():  # pylint: disable=too-few-public-methods
    """Discord configuration handler."""

    def __init__(self, config_data):
        self.is_healthy: bool = False
        self.bot_token: str = config_data["bot_token"]
        self.built_in_roles: List[str] = config_data["built_in_roles"]
        self.max_latency: float = config_data["max_latency"]


class OpenAIConfig():  # pylint: disable=too-few-public-methods
    """OpenAI configuration handler."""

    def __getitem__(self, key: str) -> str:
        return getattr(self, key)

    def __init__(self, config_data):
        self.is_healthy: bool = True
        self.filter: bool = config_data["filter"]
        self.api_key: str = config_data["api_key"]
        self.organization: str = config_data["organization"]
        self.enabled: bool = config_data["enabled"]
        self.sentiment_analysis_prompt = config_data["sentiment_analysis_prompt"]


class Config:  # pylint: disable=too-many-instance-attributes
    """Configuration handler."""

    _instances: Dict[str, 'Config'] = {}
    _file_path = os.path.join(
        os.path.curdir,
        "config.yml",
    )

    def __new__(cls):
        """Creates a new Config instance."""
        if None not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[""] = instance
        return cls._instances[""]

    def __init__(self):
        """Initializes a new Config instance."""
        if not self._initialized:
            self._initialized = True

            self.app: AppConfig
            self.api: APIConfig
            self.logger: LoggerConfig
            self.telegram: TelegramConfig
            self.discord: DiscordConfig
            self.openai: OpenAIConfig
            self.telegram_forwarders = []

            self.load()

    def set_file_path(self, version: str):
        """Set the file path based on the version."""
        if version:
            self._file_path = os.path.join(
                os.path.curdir,
                f"config-{version}.yml",
            )
        else:
            self._file_path = os.path.join(
                os.path.curdir,
                "config.yml",
            )

    def set_version(self, version: str):
        """Set the version of the configuration."""
        if version not in Config._instances:
            Config._instances[version] = Config()
        Config._instances[version].set_file_path(version)
        Config._instances[version].load()
        return Config._instances[version]

    @staticmethod
    def get_config_instance(version: str | None = "") -> 'Config':
        """Get the configuration instance based on the version."""
        try:
            return Config._instances[version] if version and version in Config._instances else Config._instances[""]
        except KeyError as ex:
            raise ValueError(f'Invalid version: {version}') from ex

    # Set the configuration instance based on the version.
    @staticmethod
    def set_config_instance(config: 'Config', version: str | None = "") -> Config:
        """Set the configuration instance based on the version."""
        if not version:
            version = ""
        if version not in Config._instances:
            print("Setting config instance for version %s", version)
            Config._instances[version] = config
        return config

    def load(self) -> Any:
        """Load configuration from the 'config-{version}.yml' file."""
        try:
            with open(self._file_path, 'rb') as config_file:
                config_data = yaml.safe_load(config_file)
        except FileNotFoundError:
            print("Error: Configuration file not found.")
            sys.exit(1)
        except yaml.YAMLError as ex:
            print("Error parsing configuration file: %s", ex)
            sys.exit(1)

        required_keys = [
            "application",
            "logger",
            "telegram",
            "discord",
            "telegram_forwarders",
        ]

        for key in required_keys:
            if key not in config_data:
                print(
                    "Error: Key %s not found in the configuration file.", key)
                sys.exit(1)

        valid, errors = Config.validate_config(config_data)

        if not valid:
            print("Error: Invalid configuration file.")
            for error in errors:
                print(f"\n{error}\n")
            sys.exit(1)

        self.app = AppConfig(config_data["application"])
        self.api = APIConfig(config_data["api"])
        self.logger = LoggerConfig(config_data["logger"])
        self.telegram = TelegramConfig(config_data["telegram"])
        self.discord = DiscordConfig(config_data["discord"])
        self.openai = OpenAIConfig(config_data["openai"])

        self.telegram_forwarders = config_data["telegram_forwarders"]

        return config_data

    @ staticmethod
    def validate_openai_enabled(config: OpenAIConfig) -> Tuple[bool, str]:
        """Check for valid types"""
        if config["enabled"]:
            if config["api_key"] == "" or config["organization"] == "" or config["sentiment_analysis_prompt"] is None:
                return False,  "Invalid configuration: `api_key`, `organization`, and `sentiment_analysis_prompt` must be set when `enabled` is True."  # pylint: disable=line-too-long

        return True, ""

    @ staticmethod
    def validate_forwarder_types(forwarder) -> Tuple[bool, str]:
        """Check for valid types"""
        tg_channel_id = forwarder["tg_channel_id"]
        discord_channel_id = forwarder["discord_channel_id"]

        if not isinstance(tg_channel_id, int):
            return False, f"Invalid configuration: `tg_channel_id` must be an integer: forwarder with `tg_channel_id` {tg_channel_id}"  # pylint: disable=line-too-long

        if not isinstance(discord_channel_id, int):
            return False, f"Invalid configuration: `discord_channel_id` must be an integer: forwarder with `tg_channel_id` {tg_channel_id}"  # pylint: disable=line-too-long

        return True, ""

    @ staticmethod
    def validate_forwarder_combinations(forwarder, forwarder_combinations) -> Tuple[bool, str]:
        """Check for unique combination of tg_channel_id and discord_channel_id"""
        tg_channel_id = forwarder["tg_channel_id"]
        discord_channel_id = forwarder["discord_channel_id"]

        combination = (tg_channel_id, discord_channel_id)
        if combination in forwarder_combinations:
            return False, f"Invalid configuration: duplicate forwarder with combination {combination}"

        forwarder_combinations.add(combination)
        return True, ""

    @ staticmethod
    def validate_mention_everyone_and_override(forwarder, forward_hashtags) -> Tuple[bool, str]:
        """Check for mention_everyone and override_mention_everyone conflict"""
        tg_channel_id = forwarder["tg_channel_id"]
        mention_everyone = forwarder["mention_everyone"]

        if mention_everyone and any(tag.get("override_mention_everyone", False) for tag in forward_hashtags):
            return False, f"Invalid configuration: `override_mention_everyone` has no effect when `mention_everyone` set to True: forwarder with `tg_channel_id` {tg_channel_id}"  # pylint: disable=line-too-long

        return True, ""

    @ staticmethod
    def validate_shared_hashtags(forwarders) -> Tuple[bool, str]:
        """Check for shared hashtags in forwarders with the same tg_channel_id"""
        tg_channel_hashtags = {}
        for forwarder in forwarders:
            tg_channel_id = forwarder["tg_channel_id"]
            forward_hashtags = {tag["name"].lower() for tag in forwarder.get(
                "forward_hashtags", [])} if forwarder.get("forward_hashtags") else set()

            if forward_hashtags:  # Only process non-empty forward_hashtags
                if tg_channel_id not in tg_channel_hashtags:
                    tg_channel_hashtags[tg_channel_id] = [forward_hashtags]
                else:
                    for existing_hashtags in tg_channel_hashtags[tg_channel_id]:
                        shared_hashtags = existing_hashtags.intersection(
                            forward_hashtags)
                        if shared_hashtags:
                            return False, f"Shared hashtags {shared_hashtags} found for forwarders with tg_channel_id {tg_channel_id}. The same message will be forwarded multiple times."  # pylint: disable=line-too-long

                    tg_channel_hashtags[tg_channel_id].append(forward_hashtags)

        return True, ""

    @staticmethod
    def validate_hashtags_overlap(forwarder, forward_hashtags, excluded_hashtags) -> Tuple[bool, str]:
        """Check for overlapping hashtags between forward_hashtags and excluded_hashtags"""
        tg_channel_id = forwarder["tg_channel_id"]
        forward_hashtags_names = {tag["name"].lower()
                                  for tag in forward_hashtags}
        excluded_hashtags_names = {tag["name"].lower()
                                   for tag in excluded_hashtags}
        common_hashtags = forward_hashtags_names.intersection(
            excluded_hashtags_names)
        if common_hashtags:
            return False, f"Invalid configuration: overlapping hashtags {common_hashtags} found in forward_hashtags and excluded_hashtags for forwarder with `tg_channel_id` {tg_channel_id}"  # pylint: disable=line-too-long

        return True, ""

    @ staticmethod
    def validate_config(config) -> Tuple[bool, List[str]]:
        """Validate the configuration."""
        forwarders = config["telegram_forwarders"]
        forwarder_combinations = set()

        valid = True
        errors: List[str] = []

        valid, error = Config.validate_openai_enabled(config["openai"])
        if not valid:
            errors.append(error)

        forwarder_error_string = "Invalid forwarder configuration:"
        for forwarder in forwarders:
            forwarder_error_string = f"{forwarder_error_string} forwarder name: {forwarder['forwarder_name']}"

            forward_hashtags = Config.get_forward_hashtags(forwarder)

            if len(forward_hashtags) <= 0 and forwarder["forward_everything"] is False:
                valid = False
                errors.append(
                    f'{forwarder_error_string} `forward_hashtags` must be set when `forward_everything` is False')  # pylint: disable=line-too-long

            excluded_hashtags = Config.get_excluded_hashtags(forwarder)

            valid, error = Config.validate_forwarder_types(forwarder)
            if not valid:
                errors.append(f"{forwarder_error_string} {error}")

            valid, error = Config.validate_forwarder_combinations(
                forwarder, forwarder_combinations)
            if not valid:
                errors.append(f"{forwarder_error_string} {error}")

            valid, error = Config.validate_mention_everyone_and_override(
                forwarder, forward_hashtags)
            if not valid:
                errors.append(f"{forwarder_error_string} {error}")

            valid, error = Config.validate_hashtags_overlap(
                forwarder, forward_hashtags, excluded_hashtags)
            if not valid:
                errors.append(f"{forwarder_error_string} {error}")

        valid, error = Config.validate_shared_hashtags(forwarders)
        if not valid:
            errors.append(error)

        valid = not errors
        return valid, errors

    @staticmethod
    def get_excluded_hashtags(forwarder):
        """Get exclude_hashtags from forwarder or set an empty list."""
        if "excluded_hashtags" in forwarder:
            excluded_hashtags = forwarder["excluded_hashtags"]
        else:
            excluded_hashtags = []

        return excluded_hashtags

    @ staticmethod
    def get_forward_hashtags(forwarder):
        """Get forward_hashtags from forwarder or set an empty list."""
        if "forward_hashtags" in forwarder:
            forward_hashtags = forwarder["forward_hashtags"]
        else:
            forward_hashtags = []

        return forward_hashtags

    def get_telegram_channel_by_forwarder_name(self, forwarder_name: str):
        """Get the Telegram channel ID associated with a given forwarder ID."""
        for forwarder in self.telegram_forwarders:
            if forwarder.get("forwarder_name") == forwarder_name:
                return forwarder.get("tg_channel_id")
        return None
