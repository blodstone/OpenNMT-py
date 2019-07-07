import re
import argparse
import logging
if __name__ == "__main__":
    logger = logging.getLogger('Experiment With Topic Matrix')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-docs', help='The input documents')
    parser.add_argument('-summ', help='The input summary')
    parser.add_argument('-output', help="The output results")
    parser.add_argument('-type', help="The output results")
    args = parser.parse_args()
    substring_weather = ['forecast', 'weather', 'temperature', 'warn']
    substring_terrorist = ['attack', 'terrorist', 'bomb', 'terror', 'qaeda']
    substring_health = ['treatment', 'drug', 'health', 'patient', 'medicine', 'disease']
    substring_finance = ['stock', 'market', 'share', 'finance']
    weather_topic = 0
    terrorist_topic = 0
    health_topic = 0
    finance_topic = 0
    if args.type == 'train':
        max_weather_topic = 70
        max_terrorist_topic = 70
        max_health_topic = 70
        max_finance_topic = 70
    elif args.type == 'validation':
        max_weather_topic = 100
        max_terrorist_topic = 100
        max_health_topic = 100
        max_finance_topic = 100
    output_lines = []
    output_idx = []
    summ_lines = open(args.summ, 'r').readlines()
    with open(args.docs, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            # print('Processed line: {}'.format(idx))
            if all(x in line for x in substring_weather) \
                    and weather_topic < max_weather_topic:
                weather_topic += 1
                # print('Found line: {} \n {} \n'.format(idx, line))
                output_lines.append(line)
                output_idx.append(idx)
            elif all(x in line for x in substring_terrorist) \
                    and terrorist_topic < max_terrorist_topic:
                terrorist_topic += 1
                # print('Found line: {} \n {} \n'.format(idx, line))
                output_lines.append(line)
                output_idx.append(idx)
            elif all(x in line for x in substring_health) \
                    and health_topic < max_health_topic:
                health_topic += 1
                # print('Found line: {} \n {} \n'.format(idx, line))
                output_lines.append(line)
                output_idx.append(idx)
            # elif all(x in line for x in substring_finance) \
            #         and finance_topic < max_finance_topic:
            #     finance_topic += 1
            #     print('Found line: {} \n {} \n'.format(idx, line))
            #     output_lines.append(line)
            #     output_idx.append(idx)
    output_file = open(args.output+'/src.'+args.type+'.select.token', 'w')
    output_file.writelines(output_lines)
    output_file.close()
    output_lines = []
    for i in output_idx:
        output_lines.append(summ_lines[i])
    output_file = open(args.output + '/tgt.' + args.type + '.select.token', 'w')
    output_file.writelines(output_lines)
    output_file.close()
    print('Weather topic: {}'.format(weather_topic))
    print('Terrorist topic: {}'.format(terrorist_topic))
    print('Health topic: {}'.format(health_topic))
    print('Finance topic: {}'.format(finance_topic))
