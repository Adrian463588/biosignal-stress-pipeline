import logging

logger = logging.getLogger(__name__)


def extract_waveforms():
    logger.info("Extracting waveforms for DL...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_waveforms()
