import logging
import shutil

from rich.logging import RichHandler

from jss_utils.banner import big_banner, small_banner

# print banner when logger is imported
w, h = shutil.get_terminal_size((80, 20))

print(small_banner if w < 140 else big_banner)

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)]
)

log = logging.getLogger("rich")

if __name__ == '__main__':
    pass
