import csv

# PURPOSE
# Write lines to a file.
# SIGNATURE
# write_lines_to_files :: List[String], String => None
def write_lines_to_file(lines, fpath):
    with open(fpath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

# PURPOSE
# Given a file path, a list of fields, and a list of lists of rows,
# write the information to a csv file.
# SIGNATURE
# write_to_csv :: String, List, List[List] => None
def write_to_csv(fpath, fields, rows):
    with open(fpath, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)