import logging
import logging.handlers


def setup_logger(log_file_path):
    # Create a logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )

    # disable certain matplotlib warnings
    # logging.getLogger('matplotlib.font_manager').disabled = True # ignore matplotlibs font warnings

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
