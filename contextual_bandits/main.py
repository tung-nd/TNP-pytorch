import time

from utils.misc import hrminsec
from runner import args
from runner.cmab_runner import cmab


def main():
    start = time.time()
    cmab(args)
    duration = int(time.time() - start)
    print(f"cmab experiment completed in {hrminsec(duration)}.")


if __name__ == "__main__":
    main()