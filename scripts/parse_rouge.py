import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str,
                        help='candidate file')
    parser.add_argument('-r', type=str,
                        help='reference file')
    parser.add_argument('-output', type=str,
                        help='reference file')
    args = parser.parse_args()
    c_f = open(args.c).readlines()
    c_str = [line.replace('<eos>','').replace('<sos>','').strip() for line in c_f]
    r_f = open(args.r).readlines()
    r_str = [line.replace('<eos>','').replace('<sos>','').strip()  for line in r_f]
    c_out = open(args.output + 'candidate.txt', 'w')
    c_out.write('\n'.join(c_str))
    r_out = open(args.output + 'reference.txt', 'w')
    r_out.write('\n'.join(r_str))
    c_out.close()
    r_out.close()
