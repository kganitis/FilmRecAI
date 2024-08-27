import datetime


class Logger:
    DEBUG = 0
    VERBOSE = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

    def __init__(self, log_level=INFO):
        self.log_level = log_level

    def log(self, message, level=INFO, filename=None, filemode='w', new_line_before=False):
        """Prints a message if the log level is less than or equal to the set log level."""
        log_levels = {self.DEBUG: "DEBUG", self.VERBOSE: "VERBOSE", self.INFO: "INFO",
                      self.WARNING: "WARNING", self.ERROR: "ERROR", self.CRITICAL: "CRITICAL"}
        if level >= self.log_level:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{log_levels.get(level, 'INFO')}] {message}"
            if new_line_before:
                log_message = f"\n{log_message}"
            if filename:
                with open(filename, filemode) as log:
                    log.write(f"[{timestamp}] {log_message}\n")
            else:
                print(log_message)
