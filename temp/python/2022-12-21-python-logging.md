---
title:  "Python logging 실습기"
toc: true
toc_sticky: true
categories:
  - Python
tags:
  - logging
use_math: true
last_modified_at: 2022-12-21
---

## Introduction

회사에서 Nvidia Triton으로 모델 inference serving을 하는 도중 로그를 쌓게되었다.


## Logger

## config를 통한 로그설정

```
%(pathname)s Full pathname of the source file where the logging call was issued(if available).

%(filename)s Filename portion of pathname.

%(module)s Module (name portion of filename).

%(funcName)s Name of function containing the logging call.

%(lineno)d Source line number where the logging call was issued (if available).
```


## 일자별로 로그를 쌓는 TimedRotatingFileHandler

```py
import logging, logging.handlers

logging.handlers.TimedRotatingFileHandler(filename=f'./log/logfile', when='midnight', interval=1, encoding='utf-8')
```

## 여러 프로세스에서 로그에 접근하여 쓰기

```
# log.ini
[loggers]
keys=root

[handlers]
keys=logfile,logconsole

[formatters]
keys=logformatter

[logger_root]
level=INFO
handlers=logfile, logconsole

[formatter_logformatter]
format=[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s

[handler_logfile]
# Class 수정
class=app.common.logger_handler.SafeRotatingFileHandler
level=INFO
args=('log','midnight')
formatter=logformatter

[handler_logconsole]
class=handlers.logging.StreamHandler
level=INFO
args=()
formatter=logformatter
```

만일 ini파일을 통해 config를 설정할 것이라면 handler로 전달하는 args에 로그파일의 위치 등을 설정해주어야 한다.

```python
import os
import time
from logging import FileHandler
from logging.handlers import TimedRotatingFileHandler

# 참고 : https://ko.n4zc.com/article/programming/python/hr8m47f3.html

class SafeRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backup_count=0, encoding=None, delay=False, utc=False):
        TimedRotatingFileHandler.__init__(self, filename, when, interval, backup_count, encoding, delay, utc)
    """
    Override doRollover
    lines commanded by "##" is changed by cc
    """
    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens. However, you want the file to be named for the
        start of the interval, not the current time. If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
            Override,  1. if dfn not exist then do rename
                2. _open with "a" model
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        current_time = int(time.time())
        dst_now = time.localtime(current_time)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            time_tuple = time.gmtime(t)
        else:
            time_tuple = time.localtime(t)
            dst_then = time_tuple[-1]
            if dst_now != dst_then:
                if dst_now:
                    addend = 3600
                else:
                    addend = -3600
                time_tuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, time_tuple)
        # if os.path.exists(dfn):
        #     os.remove(dfn)

        # Issue 18940: A file may not have been created if delay is True.
        # if os.path.exists(self.baseFilename):
        if not os.path.exists(dfn) and os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.mode = "a"
            self.stream = self._open()
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= current_time:
            new_rollover_at = new_rollover_at + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dst_at_rollover = time.localtime(new_rollover_at)[-1]
            if dst_now != dst_at_rollover:
                if not dst_now:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                new_rollover_at += addend
        self.rolloverAt = new_rollover_at
```

https://cha-vi.tistory.com/entry/Python-Uvicorn-%EC%9D%BC%EC%9E%90%EB%B3%84-%EB%A1%9C%EA%B7%B8-%EC%8C%93%EA%B8%B0with-FastAPI



{: .align-center}{: width="600"}
