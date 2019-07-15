import drmaa
import os

def main():
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = 'test.sh'
        jt.args = []
        jt.joinFiles = True



