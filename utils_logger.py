import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    return logging.getLogger()

logger = setup_logger()
