#!/bin/bash

_pid=$(pgrep -f "main.py")
kill -9 $_pid