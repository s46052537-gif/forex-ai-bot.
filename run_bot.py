"""Run the bot from the command line (no window). Stop with Ctrl+C."""
from config import load_config
from bot import TradingBot


def main():
    cfg = load_config()
    bot = TradingBot(cfg)
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
        print("Stopped by user.")


if __name__ == "__main__":
    main()
