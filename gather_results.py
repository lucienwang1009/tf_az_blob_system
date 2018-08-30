import logging
import os
import re
import glob
import csv

logger = logging.getLogger(name=__name__)

def fetch_log(log_file):
    if not os.path.exists(log_file):
        raise RuntimeError('Cannot find the %s log file!' % log_file)
    log = None
    with open(log_file, 'r', encoding='utf8') as f:
        log = f.read()
    return log

def fetch_throughput_from_log(train_log_file,
                              warmup_steps=1,
                              batch_size=32):
    log = fetch_log(train_log_file)
    log_lines = log.splitlines()
    total_speed = 0
    cnt = 0
    warmup_skip = True
    for l in log_lines:
        m = re.findall(r'step = (\d+)', l)
        if len(m) > 0 and int(m[0]) >= warmup_steps:
            warmup_skip = False
        m = re.findall(r'global_step/sec:\s*([\d\.]*)', l)
        if len(m) > 0 and not warmup_skip:
            total_speed += float(m[0])
            cnt += 1
    if cnt == 0:
        raise RuntimeError('Cannot find the throughput in the training log: %s' % train_log_file)
    return total_speed * batch_size / cnt

def parse_filename(file_name):
    return os.path.splitext(file_name)[0]

def save_results(file_name, results):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        for k, v in results:
            writer.writerow([k, v])

def main():
    results = {}
    results_files = glob.glob('*.stderr')
    for log_file in results_files:
        throughput = fetch_throughput_from_log(log_file)
        results[parse_filename(log_file)] = throughput
    save_results(results, './results.csv')

if __name__ == '__main__':
    main()
