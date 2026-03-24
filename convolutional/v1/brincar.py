#!/usr/bin/env python3
from time import sleep

if __name__ == '__main__':
    count =0
    for i in range(0,999999999):
        count += 1
        sleep(1)
        print(f"count: {count}")