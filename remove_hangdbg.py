import shutil, os

filepath = os.environ.get(
    "DYME_LOG_PATH",
    "outputs/test-fast/logs/train_test_opd_20260621_212323.log",
)
tmpfile = filepath + '.tmp'
count = 0
total = 0

with open(filepath, 'r') as fin, open(tmpfile, 'w') as fout:
    for line in fin:
        total += 1
        if 'OPSD-HANGDBG' not in line:
            fout.write(line)
        else:
            count += 1

shutil.move(tmpfile, filepath)
print(f'Done. Removed {count} lines out of {total} total lines.')
