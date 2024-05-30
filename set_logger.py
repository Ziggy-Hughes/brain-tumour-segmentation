import os
import time
import logging
#from raise_error import RaiseError as Error

def get_log_file_path():
    log_folder = 'LogFolder'
    log_abspath = os.path.join(os.path.abspath('..'), log_folder)
    if not os.path.exists(log_abspath):
        os.makedirs(log_abspath)
    return log_abspath

class Setlogger(object):
    def __init__(self):
        #print("********* Set The Configuration of Logger ********* \n")
        self.log_abspath = get_log_file_path()

    def create_logger(self):
        log_file_name = 'LOG_' + time.strftime('%Y%m%d_%H-%M-%S',time.localtime(time.time())) + '.log'
        log_file_path = os.path.join(self.log_abspath, log_file_name)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d_%H:%M:%S',
                            filename=log_file_path,
                            filemode='w')
        # create logger
        self.logger = logging.getLogger('root')
        self.logger.setLevel(logging.INFO)
        # create console handler
        self.ch_file = logging.FileHandler(log_file_path)
        self.ch_stream = logging.StreamHandler()
        # set logger debug level
        self.ch_file.setLevel(logging.INFO)
        self.ch_stream.setLevel(logging.INFO)
        # create formatter
        formatter = logging.Formatter('[%(asctime)s-%(name)s-%(levelname)s] %(message)s')
        # add formatter to ch
        self.ch_file.setFormatter(formatter)
        self.ch_stream.setFormatter(formatter)
        # add ch to self.logger
        self.logger.addHandler(self.ch_file)
        self.logger.addHandler(self.ch_stream)
        return self.logger

    def remove_logger(self):
        self.logger.removeHandler(self.ch_file)
        self.logger.removeHandler(self.ch_stream)
