import argparse
import os


def remove_sos_eos(file):
    i = 1
    result = []
    for line in file:
        print(i)
        i += 1
        l_result = []
        lines = line.split()
        for l in lines:
            if l == '<sos>':
                continue
            if l == '<eos>':
                break
            l_result.append(l)
        result.append(' '.join(l_result))
    return result

def remove_sos_eos_lines(alist):
    result = []
    for line in alist:
        line = line.strip()
        l_result = []
        lines = line.split()
        for l in lines:
            if l == '<sos>':
                continue
            if l == '<eos>':
                continue
            l_result.append(l)
        result.append(' '.join(l_result))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str,
                        help='candidate file')
    parser.add_argument('-r', type=str,
                        help='reference file')
    parser.add_argument('-output_dir', type=str,
                        help='reference file')
    parser.add_argument('-output_name', type=str,
                        help='output name')
    args = parser.parse_args()
    c_f = open(args.c).readlines()
    r_f = open(args.r).readlines()
    print('Cleaning candidate file.')
    c_out = open(os.path.join(args.output_dir, args.output_name + '.c.txt'), 'w')
    c_out.write('\n'.join(remove_sos_eos(c_f)))
    print('Cleaning reference file.')
    r_out = open(os.path.join(args.output_dir, args.output_name + '.r.txt'), 'w')
    r_out.write('\n'.join(remove_sos_eos(r_f)))
    c_out.close()
    r_out.close()
