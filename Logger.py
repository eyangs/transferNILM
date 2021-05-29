import logging
import time

def create_logfile():
    log_file_name = '{}.log'.format(time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))

    with open(log_file_name, 'w'):
        pass

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.DEBUG)
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler("{0}".format(log_file_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


def log(string, level='info'):
    rootLogger = create_logfile()

    if level == 'info':
        rootLogger.info(string)
    elif level == 'debug':
        rootLogger.debug(string)
    elif level == 'warning':
        rootLogger.warning(string)

